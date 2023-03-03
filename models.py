import math

import flax.linen as nn
import jax
import jax.numpy as jnp

import attentions
import commons
import modules

from modules import FlaxConvWithWeightNorm, FlaxConvTransposeWithWeightNorm


class StochasticDurationPredictor(nn.Module):
    in_channels: int
    filter_channels: int
    kernel_size: int
    p_dropout: float
    n_flows: int = 4
    gin_channels: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.log_flow = modules.Log()

        flows = []
        flows.append(modules.ElementwiseAffine(2, dtype=self.dtype))
        for i in range(self.n_flows):
            flows.append(
                modules.ConvFlow(
                    2, self.filter_channels, self.kernel_size, 3, dtype=self.dtype
                )
            )
            flows.append(modules.Flip())
        self.flows = flows

        self.post_pre = nn.Conv(self.filter_channels, (1,), dtype=self.dtype)
        self.post_proj = nn.Conv(self.filter_channels, (1,), dtype=self.dtype)
        self.post_convs = modules.DDSConv(
            self.filter_channels, self.kernel_size, 3, self.p_dropout
        )

        post_flows = []
        post_flows.append(modules.ElementwiseAffine(2, dtype=self.dtype))
        for i in range(self.n_flows):
            post_flows.append(
                modules.ConvFlow(
                    2, self.filter_channels, self.kernel_size, 3, dtype=self.dtype
                )
            )
            post_flows.append(modules.Flip())
        self.post_flows = post_flows

        self.pre = nn.Conv(self.filter_channels, (1,), dtype=self.dtype)
        self.proj = nn.Conv(self.filter_channels, (1,), dtype=self.dtype)
        self.convs = modules.DDSConv(
            self.filter_channels, self.kernel_size, 3, self.p_dropout
        )
        if self.gin_channels > 0:
            self.cond = nn.Conv(self.filter_channels, (1,), dtype=self.dtype)

    def __call__(
        self,
        x,
        x_mask,
        w=None,
        g=None,
        reverse: bool = False,
        noise_scale: float = 1.0,
        deterministic: bool = True,
    ):
        x = jax.lax.stop_gradient(x)
        x = self.pre(x)
        if g is not None:
            g = jax.lax.stop_gradient(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask, deterministic)
        x = self.proj(x) * x_mask

        if not reverse:
            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = (
                jax.random.normal(
                    self.make_rng("normal"),
                    (w.shape[0], w.shape[2], 2),
                    dtype=self.dtype,
                )
                * x_mask
            )
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(
                    z_q, x_mask, g=(x + h_w), deterministic=deterministic
                )
                logdet_tot_q += logdet_q
            z_u, z1 = jnp.split(z_q, 2, axis=2)
            u = nn.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += jnp.sum(
                nn.log_sigmoid(z_u) + nn.log_sigmoid(-z_u) * x_mask, axis=(1, 2)
            )
            logq = (
                jnp.sum(-0.5 * (jnp.log(2 * math.pi) + z1**2) * x_mask, axis=(1, 2))
                - logdet_q
            )

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = jnp.concatenate([z0, z1], axis=2)
            for flow in self.flows:
                z, logdet = flow(
                    z, x_mask, g=x, reverse=reverse, deterministic=deterministic
                )
                logdet_tot += logdet
            nll = (
                jnp.sum(-0.5 * (jnp.log(2 * math.pi) + z**2) * x_mask, axis=(1, 2))
                - logdet_tot
            )
            return nll + logq
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]
            z = (
                jax.random.normal(
                    self.make_rng("normal"),
                    (x.shape[0], 2, x.shape[2]),
                    dtype=self.dtype,
                )
                * noise_scale
            )
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse, deterministic=deterministic)
            z0, z1 = jnp.split(z, 2, axis=2)
            logw = z0
            return logw


class TextEncoder(nn.Module):
    n_vocab: int
    out_channels: int
    hidden_channels: int
    filter_channels: int
    n_heads: int
    n_layers: int
    kernel_size: int = 1
    p_dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.emb = nn.Embed(
            self.n_vocab,
            self.hidden_channels,
            dtype=self.dtype,
        )

        self.encoder = attentions.Encoder(
            self.hidden_channels,
            self.filter_channels,
            self.n_heads,
            self.n_layers,
            self.kernel_size,
            self.p_dropout,
            dtype=self.dtype,
        )
        self.proj = nn.Conv(
            self.out_channels * 2,
            (1,),
            dtype=self.dtype,
        )

    def __call__(self, x, x_lengths, deterministic: bool = True):
        x = self.emb(x) * math.sqrt(self.hidden_channels)
        x_mask = jnp.arange(x.shape[1]) < x_lengths[:, None]
        x_mask = jnp.expand_dims(x_mask, axis=2).astype(self.dtype)

        x = self.encoder(
            x * x_mask, x_mask.transpose(0, 2, 1), deterministic=deterministic
        )
        stats = self.proj(x) * x_mask

        m, logs = jnp.split(stats, 2, axis=2)
        return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    channels: int
    hidden_channels: int
    kernel_size: int
    dilation_rate: int
    n_layers: int
    n_flows: int = 4
    gin_channels: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        flows = []
        for i in range(self.n_flows):
            flows.append(
                modules.ResidualCouplingLayer(
                    self.channels,
                    self.hidden_channels,
                    self.kernel_size,
                    self.dilation_rate,
                    self.n_layers,
                    gin_channels=self.gin_channels,
                    dtype=self.dtype,
                )
            )
            flows.append(modules.Flip())
        self.flows = flows

    def __call__(self, x, x_mask, g=None, reverse: bool = False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    in_channels: int
    out_channels: int
    hidden_channels: int
    kernel_size: int
    dilation_rate: int
    n_layers: int
    gin_channels: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.pre = nn.Conv(self.hidden_channels, (1,), dtype=self.dtype)
        self.enc = modules.WN(
            self.hidden_channels,
            self.kernel_size,
            self.dilation_rate,
            self.n_layers,
            self.gin_channels,
            dtype=self.dtype,
        )
        self.proj = nn.Conv(self.out_channels * 2, (1,), dtype=self.dtype)

    def __call__(self, x, x_lengths, g=None, deterministic: bool = True):
        x_mask = jnp.arange(x.shape[1]) < x_lengths[:, None]
        x_mask = jnp.expand_dims(x_mask, axis=2).astype(self.dtype)

        x = self.pre(x) * x_mask
        x = self.enc(x, x_lengths, g, deterministic)
        stats = self.proj(x) * x_mask
        m, logs = jnp.split(stats, 2, axis=2)

        z = (
            m
            + jnp.exp(logs)
            * jax.random.normal(self.make_rng("normal"), m.shape, dtype=self.dtype)
        ) * x_mask
        return z, m, logs, x_mask


class Generator(nn.Module):
    initial_channel: int
    resblock_kernel_sizes: list
    resblock_dilation_sizes: list
    upsample_rates: tuple
    upsample_initial_channel: int
    upsample_kernel_sizes: list
    gin_channels: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.num_kernels = len(self.resblock_kernel_sizes)
        self.num_upsamples = len(self.upsample_rates)

        self.conv_pre = nn.Conv(self.upsample_initial_channel, (7,), 1, padding=3)

        self.ups = [
            FlaxConvTransposeWithWeightNorm(
                self.upsample_initial_channel // (2 ** i),
                self.upsample_initial_channel // (2 ** (i + 1)),
                (k,),
                (u,),
                dtype=self.dtype,
            )
            for i, (u, k) in enumerate(
                zip(self.upsample_rates, self.upsample_kernel_sizes)
            )
        ]

        resblocks = []
        for i in range(len(self.ups)):
            ch = self.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(self.resblock_kernel_sizes, self.resblock_dilation_sizes)
            ):
                resblocks.append(modules.ResBlock1(ch, k, d, dtype=self.dtype))
        self.resblocks = resblocks

        self.conv_post = nn.Conv(1, (7,), (1,), use_bias=False)

        if self.gin_channels != 0:
            self.cond = nn.Conv(self.upsample_initial_channel, 1)

    def __call__(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = nn.leaky_relu(x)
        x = self.conv_post(x)
        x = nn.tanh(x)

        return x


class DiscriminatorP(nn.Module):
    period: int
    kernel_size: int = 5
    stride: int = 3
    use_spectral_norm: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.use_spectral_norm:
            raise NotImplementedError

        self.convs = [
            FlaxConvWithWeightNorm(1, 32, (self.kernel_size, 1), (self.stride, 1)),
            FlaxConvWithWeightNorm(32, 128, (self.kernel_size, 1), (self.stride, 1)),
            FlaxConvWithWeightNorm(128, 512, (self.kernel_size, 1), (self.stride, 1)),
            FlaxConvWithWeightNorm(512, 1024, (self.kernel_size, 1), (self.stride, 1)),
            FlaxConvWithWeightNorm(1024, 1024, (self.kernel_size, 1), 1),
        ]
        self.conv_post = FlaxConvWithWeightNorm(1024, 1, (3, 1), 1)

    def __call__(self, x):
        fmap = []

        b, t, c = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = jnp.pad(x, ((0, 0), (0, n_pad), (0, 0)), mode="reflect")
            t = t + n_pad
        x = x.reshape(b, t // self.period, self.period, c)

        for l in self.convs:
            x = l(x)
            x = nn.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = x.reshape(1, -1)
        return x, fmap


class DiscriminatorS(nn.Module):
    use_spectral_norm: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.use_spectral_norm:
            raise NotImplementedError

        self.convs = [
            FlaxConvWithWeightNorm(1, 16, (15,), 1),
            FlaxConvWithWeightNorm(16, 64, (41,), 4, feature_group_count=4),
            FlaxConvWithWeightNorm(64, 256, (41,), 4, feature_group_count=16),
            FlaxConvWithWeightNorm(256, 1024, (41,), 4, feature_group_count=64),
            FlaxConvWithWeightNorm(1024, 1024, (41,), 4, feature_group_count=256),
            FlaxConvWithWeightNorm(1024, 1024, (5,), 1),
        ]
        self.conv_post = FlaxConvWithWeightNorm(1024, 1, (3,), 1)

    def __call__(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = nn.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = x.reshape(1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    use_spectral_norm: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.use_spectral_norm:
            raise NotImplementedError

        periods = [2, 3, 5, 7, 11]

        self.discriminators = [
            DiscriminatorS(use_spectral_norm=self.use_spectral_norm)
        ] + [
            DiscriminatorP(period, use_spectral_norm=self.use_spectral_norm)
            for period in periods
        ]

    def __call__(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SynthesizerTrn(nn.Module):
    n_vocab: int
    spec_channels: int
    segment_size: int
    inter_channels: int
    hidden_channels: int
    filter_channels: int
    n_heads: int
    n_layers: int
    kernel_size: int
    p_dropout: int
    resblock_kernel_sizes: int
    resblock_dilation_sizes: int
    upsample_rates: int
    upsample_initial_channel: int
    upsample_kernel_sizes: int
    n_speakers: int = 1
    gin_channels: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.enc_p = TextEncoder(
            self.n_vocab,
            self.inter_channels,
            self.hidden_channels,
            self.filter_channels,
            self.n_heads,
            self.n_layers,
            self.kernel_size,
            self.p_dropout,
            self.dtype,
        )
        self.dec = Generator(
            self.inter_channels,
            self.resblock_kernel_sizes,
            self.resblock_dilation_sizes,
            self.upsample_rates,
            self.upsample_initial_channel,
            self.upsample_kernel_sizes,
            self.gin_channels,
            self.dtype,
        )
        self.enc_q = PosteriorEncoder(
            self.spec_channels,
            self.inter_channels,
            self.hidden_channels,
            5,
            1,
            16,
            gin_channels=self.gin_channels,
            dtype=self.dtype,
        )
        self.flow = ResidualCouplingBlock(
            self.inter_channels,
            self.hidden_channels,
            5,
            1,
            4,
            gin_channels=self.gin_channels,
            dtype=self.dtype,
        )

        self.dp = StochasticDurationPredictor(
            self.hidden_channels,
            192,
            3,
            0.5,
            4,
            gin_channels=self.gin_channels,
            dtype=self.dtype,
        )

        if self.n_speakers > 1:
            self.emb_g = nn.Embed(self.n_speakers, self.gin_channels)

    def __call__(
        self, x, x_lengths, y, y_lengths, sid=None, deterministic: bool = True
    ):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, deterministic=deterministic)
        if self.n_speakers > 0:
            g = jnp.expand_dims(self.emb_g(sid), -1)
        else:
            g = None

        z, m_q, logs_q, y_mask = self.enc_q(
            y, y_lengths, g=g, deterministic=deterministic
        )
        z_p = self.flow(z, y_mask, g=g)

        s_p_sq_r = jnp.exp(-2 * logs_p)
        neg_cent1 = jnp.sum(
            -0.5 * math.log(2 * math.pi) - logs_p, axis=2, keepdims=True
        ).transpose(0, 2, 1)
        neg_cent2 = jnp.matmul(-0.5 * (z_p**2), s_p_sq_r.transpose(0, 2, 1))
        neg_cent3 = jnp.matmul(z_p, (m_p * s_p_sq_r).transpose(0, 2, 1))
        neg_cent4 = jnp.sum(
            -0.5 * (m_p**2) * s_p_sq_r, axis=2, keepdims=True
        ).transpose(0, 2, 1)
        neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

        attn_mask = jnp.expand_dims(x_mask, 1) * jnp.expand_dims(y_mask, 2)
        attn = commons.maximum_path_jax(neg_cent, attn_mask.squeeze(-1))
        attn = jnp.expand_dims(attn, 1)
        attn = jax.lax.stop_gradient(attn)

        w = attn.sum(2).transpose(0, 2, 1)
        l_length = self.dp(x, x_mask, w, g=g, deterministic=deterministic)
        l_length = l_length / jnp.sum(x_mask)

        m_p = jnp.matmul(attn.squeeze(1), m_p)
        logs_p = jnp.matmul(attn.squeeze(1), logs_p)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        o = self.dec(z_slice, g=g)

        return (
            o,
            l_length,
            attn,
            ids_slice,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
        )
