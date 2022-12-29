import flax.linen as nn
import jax.numpy as jnp


class DiscriminatorP(nn.Module):
    period: int
    kernel_size: int = 5
    stride: int = 3
    use_spectral_norm: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.convs = [
            nn.Conv(32, (self.kernel_size, 1), (self.stride, 1)),
            nn.Conv(128, (self.kernel_size, 1), (self.stride, 1)),
            nn.Conv(512, (self.kernel_size, 1), (self.stride, 1)),
            nn.Conv(1024, (self.kernel_size, 1), (self.stride, 1)),
            nn.Conv(1024, (self.kernel_size, 1), 1),
        ]
        self.conv_post = nn.Conv(1, (3, 1), 1)

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
        self.convs = [
            nn.Conv(16, (15,), 1),
            nn.Conv(64, (41,), 4, feature_group_count=4),
            nn.Conv(256, (41,), 4, feature_group_count=16),
            nn.Conv(1024, (41,), 4, feature_group_count=64),
            nn.Conv(1024, (41,), 4, feature_group_count=256),
            nn.Conv(1024, (5,), 1),
        ]
        self.conv_post = nn.Conv(1, (3,), 1)

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
        periods = [2, 3, 5, 7, 11]

        self.discriminators = [DiscriminatorS(use_spectral_norm=self.use_spectral_norm)] + \
                              [DiscriminatorP(period, use_spectral_norm=self.use_spectral_norm) for period in periods]

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
