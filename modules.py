import math
from typing import Tuple, Sequence, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from transforms import piecewise_rational_quadratic_transform


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = jnp.tanh(in_act[:, :, :n_channels_int])
    s_act = jax.nn.sigmoid(in_act[:, :, n_channels_int:])
    acts = t_act * s_act
    return acts


class FlaxConvWithWeightNorm(nn.Module):
    in_features: int
    out_features: int
    kernel_size: Sequence[int]
    strides: int = 1
    padding: Union[str, int, Sequence[Union[int, Tuple[int, int]]]] = "SAME"
    feature_group_count: int = 1
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = nn.Conv(
            features=self.out_features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            kernel_init=jax.nn.initializers.he_normal(),
            padding=self.padding,
            feature_group_count=self.feature_group_count,
            dtype=self.dtype,
        )
        weight_shape = self.kernel_size + (
            self.in_features // self.feature_group_count,
            self.out_features,
        )
        self.weight_v = self.param(
            "weight_v", jax.nn.initializers.he_normal(), weight_shape
        )
        self.weight_g = self.param(
            "weight_g",
            lambda _: jnp.linalg.norm(self.weight_v, axis=(0, 1))[None, None, :],
        )
        self.bias = self.param("bias", jax.nn.initializers.zeros, (self.conv.features,))

    def _get_normed_weights(self):
        weight_v_norm = jnp.linalg.norm(self.weight_v, axis=(0, 1))[None, None, :]
        normed_weight_v = jnp.divide(self.weight_v, weight_v_norm)
        normed_kernel = jnp.multiply(normed_weight_v, self.weight_g)
        return normed_kernel

    def __call__(self, hidden_states):
        kernel = self._get_normed_weights()
        hidden_states = self.conv.apply(
            {"params": {"kernel": kernel, "bias": self.bias}}, hidden_states
        )
        return hidden_states


class DDSConv(nn.Module):
    channels: int
    kernel_size: int
    n_layers: int
    p_dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dropout = nn.Dropout(rate=self.p_dropout)

        convs_sep = []
        convs_1x1 = []
        norms_1 = []
        norms_2 = []
        for i in range(self.n_layers):
            dilation = self.kernel_size**i
            padding = (self.kernel_size * dilation - dilation) // 2
            convs_sep.append(
                nn.Conv(
                    self.channels,
                    (self.kernel_size,),
                    padding=padding,
                    kernel_dilation=dilation,
                    feature_group_count=self.channels,
                    dtype=self.dtype,
                )
            )
            convs_1x1.append(nn.Conv(self.channels, (1,), dtype=self.dtype))
            norms_1.append(nn.LayerNorm(self.channels))
            norms_2.append(nn.LayerNorm(self.channels))

        self.convs_sep = convs_sep
        self.convs_1x1 = convs_1x1
        self.norms_1 = norms_1
        self.norms_2 = norms_2

    def __call__(self, x, x_mask, g=None, deterministic: bool = True):
        if g is not None:
            x = x + g

        for conv, ln_1, conv_1x1, ln_2 in zip(
            self.convs_sep, self.norms_1, self.convs_1x1, self.norms_2
        ):
            r = x

            x = x * x_mask
            x = nn.gelu(ln_1(conv(x)), approximate=False)
            x = nn.gelu(ln_2(conv_1x1(x)), approximate=False)
            x = self.dropout(x, deterministic=deterministic)

            x = x + r

        return x * x_mask


class WN(nn.Module):
    hidden_channels: int
    kernel_size: int
    dilation_rate: int
    n_layers: int = 6
    gin_channels: int = 0
    p_dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        in_layers = []
        res_skip_layers = []

        self.dropout = nn.Dropout(rate=self.p_dropout)

        if self.gin_channels != 0:
            self.cond_layer = nn.Conv(self.hidden_channels * 2 * self.n_layers, (1,))

        for i in range(self.n_layers):
            dilation = self.dilation_rate**i
            padding = (self.kernel_size * dilation - dilation) // 2
            in_layers.append(
                nn.Conv(
                    self.hidden_channels * 2,
                    (self.kernel_size,),
                    padding=padding,
                    kernel_dilation=dilation,
                    dtype=self.dtype,
                )
            )

            if i < self.n_layers - 1:
                res_skip_channels = 2 * self.hidden_channels
            else:
                res_skip_channels = self.hidden_channels
            res_skip_layers.append(nn.Conv(res_skip_channels, (1,), dtype=self.dtype))

        self.in_layers = in_layers
        self.res_skip_layers = res_skip_layers

    def __call__(self, x, x_mask, g=None, deterministic: bool = True):
        output = jnp.zeros_like(x)
        n_channels_tensor = jnp.array([self.hidden_channels], dtype=jnp.int32)

        if g is not None:
            g = self.cond_layer(g)

        for i, (in_layer, res_skip_layer) in enumerate(
            zip(self.in_layers, self.res_skip_layers)
        ):
            x_in = in_layer(x)

            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, :, cond_offset : cond_offset + 2 * self.hidden_channels]
            else:
                g_l = jnp.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.dropout(acts, deterministic=deterministic)

            res_skip_acts = res_skip_layer(acts)

            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :, : self.hidden_channels]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, :, self.hidden_channels :]
            else:
                output = output + res_skip_acts

        return output


class ResBlock1(nn.Module):
    channels: int
    kernel_size: int = 3
    dilations: Tuple[int] = (1, 3, 5)
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.convs1 = [
            nn.Conv(
                self.channels,
                (self.kernel_size,),
                kernel_dilation=dilation,
                dtype=self.dtype,
            )
            for dilation in self.dilations
        ]
        self.convs2 = [
            nn.Conv(
                self.channels,
                (self.kernel_size,),
                kernel_dilation=1,
                dtype=self.dtype,
            )
            for _ in self.dilations
        ]

    def __call__(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            r = x

            x = nn.leaky_relu(x, 0.1)
            x = c1(x)

            x = nn.leaky_relu(x, 0.1)
            x = c2(x)

            x = x + r

        return x


class Log(nn.Module):
    def __call__(self, x, x_mask, reverse=False):
        if not reverse:
            y = jnp.log(jnp.clip(x, a_min=1e-5, a_max=None)) * x_mask
            logdet = jnp.sum(-y, axis=(1, 2))
            return y, logdet
        else:
            x = jnp.exp(x) * x_mask
            return x


class Flip(nn.Module):
    def __call__(self, x, x_mask, reverse=False, **kwargs):
        x = jnp.flip(x, axis=2)
        if not reverse:
            logdet = jnp.zeros(x.shape[0]).astype(x.dtype)
            return x, logdet
        else:
            return x


class ElementwiseAffine(nn.Module):
    channels: int

    def setup(self):
        self.m = self.param("m", jax.nn.initializers.zeros, (1, self.channels))
        self.logs = self.param("logs", jax.nn.initializers.zeros, (1, self.channels))

    def __call__(self, x, x_mask, reverse: bool = False):
        if not reverse:
            y = self.m + jnp.exp(self.logs) * x
            y = y * x_mask
            logdet = jnp.sum(self.logs * x_mask, axis=(1, 2))
            return y, logdet
        else:
            x = (x - self.m) * jnp.exp(-self.logs) * x_mask
            return x


class ResidualCouplingLayer(nn.Module):
    channels: int
    hidden_channels: int
    kernel_size: int
    dilation_rate: int
    n_layers: int
    p_dropout: float = 0.0
    gin_channels: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        half_channels = self.channels // 2

        self.pre = nn.Conv(self.hidden_channels, (1,), dtype=self.dtype)
        self.enc = WN(
            self.hidden_channels,
            self.kernel_size,
            self.dilation_rate,
            self.n_layers,
            self.gin_channels,
            self.p_dropout,
            self.dtype,
        )
        self.post = nn.Conv(half_channels, (1,), dtype=self.dtype)

    def __call__(self, x, x_mask, g=None, reverse=False, deterministic: bool = True):
        x0, x1 = jnp.split(x, 2, axis=2)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g, deterministic=deterministic)
        stats = self.post(h) * x_mask

        m = stats
        logs = jnp.zeros_like(m)

        if not reverse:
            x1 = m + x1 * jnp.exp(logs) * x_mask
            x = jnp.concatenate([x0, x1], axis=2)
            logdet = jnp.sum(logs, (1, 2))
            return x, logdet
        else:
            x1 = (x1 - m) * jnp.exp(-logs) * x_mask
            x = jnp.concatenate([x0, x1], axis=2)
            return x


class ConvFlow(nn.Module):
    in_channels: int
    filter_channels: int
    kernel_size: int
    n_layers: int
    num_bins: int = 10
    tail_bound: int = 5.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        half_channels = self.in_channels // 2

        self.pre = nn.Conv(self.filter_channels, (1,), dtype=self.dtype)
        self.convs = DDSConv(
            self.filter_channels, self.kernel_size, self.n_layers, 0.0, self.dtype
        )
        self.proj = nn.Conv(
            half_channels * (self.num_bins * 3 - 1), (1,), dtype=self.dtype
        )

    def __call__(self, x, x_mask, g=None, reverse=False, deterministic: bool = True):
        x0, x1 = jnp.split(x, 2, axis=2)
        h = self.pre(x0)
        h = self.convs(h, x_mask, g, deterministic=deterministic)
        h = self.proj(h) * x_mask

        b, t, c = x0.shape
        h = jnp.reshape(h, (b, t, c, -1)).transpose(0, 2, 1, 3)

        unnormalized_widths = h[..., : self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / math.sqrt(
            self.filter_channels
        )
        unnormalized_derivatives = h[..., 2 * self.num_bins :]

        x1, logabsdet = piecewise_rational_quadratic_transform(
            x1.transpose(0, 2, 1),
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )
        x1 = x1.transpose(0, 2, 1)
        logabsdet = logabsdet.transpose(0, 2, 1)

        x = jnp.concatenate([x0, x1], 2) * x_mask
        logdet = jnp.sum(logabsdet * x_mask, (1, 2))
        if not reverse:
            return x, logdet
        else:
            return x
