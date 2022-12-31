import math

import flax.linen as nn
import jax.numpy as jnp


class Encoder(nn.Module):
    hidden_channels: int
    filter_channels: int
    n_heads: int
    n_layers: int
    kernel_size: int = 1
    p_dropout: float = 0.0
    window_size: int = 4
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.drop = nn.Dropout(rate=self.p_dropout)

        attn_layers = []
        norm_layers_1 = []
        ffn_layers = []
        norm_layers_2 = []
        for i in range(self.n_layers):
            attn_layers.append(
                MultiHeadAttention(
                    self.hidden_channels,
                    self.hidden_channels,
                    self.n_heads,
                    self.p_dropout,
                    self.window_size,
                    dtype=self.dtype,
                )
            )
            norm_layers_1.append(nn.LayerNorm(self.hidden_channels))
            ffn_layers.append(
                FFN(
                    self.hidden_channels,
                    self.hidden_channels,
                    self.filter_channels,
                    self.kernel_size,
                    self.p_dropout,
                    dtype=self.dtype,
                )
            )
            norm_layers_2.append(nn.LayerNorm(self.hidden_channels))

        self.attn_layers = attn_layers
        self.norm_layers_1 = norm_layers_1
        self.ffn_layers = ffn_layers
        self.norm_layers_2 = norm_layers_2

    def __call__(self, x, x_mask, deterministic: bool = True):
        attn_mask = jnp.expand_dims(x_mask, axis=2) * jnp.expand_dims(x_mask, axis=-1)
        x_mask = x_mask.transpose(0, 2, 1)

        x = x * x_mask
        for attn, norm_1, ffn, norm_2 in zip(
                self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2
        ):
            y = attn(x, x, attn_mask=attn_mask, deterministic=deterministic)
            y = self.drop(y, deterministic=deterministic)
            x = norm_1(x + y)

            y = ffn(x, x_mask)
            y = self.drop(y, deterministic=deterministic)
            x = norm_2(x + y)

        x = x * x_mask
        return x


class MultiHeadAttention(nn.Module):
    channels: int
    out_channels: int
    n_heads: int
    p_dropout: float = 0.0
    window_size: bool = None
    heads_share: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.k_channels = self.channels // self.n_heads
        self.conv_q = nn.Conv(self.channels, (1,), dtype=self.dtype)
        self.conv_k = nn.Conv(self.channels, (1,), dtype=self.dtype)
        self.conv_v = nn.Conv(self.channels, (1,), dtype=self.dtype)
        self.conv_o = nn.Conv(self.out_channels, (1,), dtype=self.dtype)
        self.drop = nn.Dropout(rate=self.p_dropout)

        if self.window_size is not None:
            n_heads_rel = 1 if self.heads_share else self.n_heads
            rel_stddev = self.k_channels ** -0.5
            self.emb_rel_k = self.param(
                "emb_rel_k",
                nn.initializers.normal(stddev=rel_stddev),
                (n_heads_rel, self.window_size * 2 + 1, self.k_channels),
            )
            self.emb_rel_v = self.param(
                "emb_rel_v",
                nn.initializers.normal(stddev=rel_stddev),
                (n_heads_rel, self.window_size * 2 + 1, self.k_channels),
            )

    def __call__(self, x, c, attn_mask=None, deterministic: bool = True):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x = self.attention(q, k, v, attn_mask, deterministic=deterministic)

        x = self.conv_o(x)

        return x

    def attention(self, query, key, value, mask=None, deterministic: bool = True):
        b, t_s, d, t_t = (*key.shape, query.shape[1])
        query = query.reshape(b, t_t, self.n_heads, self.k_channels).transpose(0, 2, 1, 3)
        key = key.reshape(b, t_s, self.n_heads, self.k_channels).transpose(0, 2, 1, 3)
        value = value.reshape(b, t_s, self.n_heads, self.k_channels).transpose(0, 2, 1, 3)

        scores = jnp.matmul(
            query / math.sqrt(self.k_channels), key.transpose(0, 1, 3, 2)
        )
        if self.window_size is not None:
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query / math.sqrt(self.k_channels), key_relative_embeddings
            )
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores += scores_local
        if mask is not None:
            scores = jnp.where(mask == 0, -1e4, scores)
        p_attn = nn.softmax(scores, axis=-1)
        p_attn = self.drop(p_attn, deterministic=deterministic)
        output = jnp.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_v, t_s
            )
            output += self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings
            )

        output = output.transpose(0, 2, 1, 3).reshape(b, t_t, d)
        return output

    def _matmul_with_relative_values(self, x, y):
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = jnp.matmul(x, jnp.expand_dims(y, axis=0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = jnp.matmul(x, jnp.expand_dims(y.transpose(0, 2, 1), axis=0))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        max_relative_position = 2 * self.window_size + 1
        pad_length = max(0, length - (self.window_size + 1))
        slice_start_position = max(0, self.window_size + 1 - length)
        slice_end_position = max_relative_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = jnp.pad(
                relative_embeddings,
                ((0, 0), (pad_length, pad_length), (0, 0)),
            )
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[
                                   :, slice_start_position:slice_end_position
                                   ]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.shape
        x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, 1)))

        x_flat = x.reshape(batch, heads, length * 2 * length)
        x_flat = jnp.pad(x_flat, ((0, 0), (0, 0), (0, length - 1)))

        x_final = x_flat.reshape(batch, heads, length + 1, 2 * length - 1)[
                  ..., :length, length - 1:
                  ]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.shape
        x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, length - 1)))
        x_flat = x.reshape(batch, heads, -1)
        x_flat = jnp.pad(x_flat, ((0, 0), (0, 0), (length, 0)))
        x_final = x_flat.reshape(batch, heads, length, 2 * length)[..., 1:]
        return x_final


class FFN(nn.Module):
    in_channels: int
    out_channels: int
    filter_channels: int
    kernel_size: int = 1
    p_dropout: float = 0.0
    activation: str = None
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv1 = nn.Conv(
            self.filter_channels, (self.kernel_size,), dtype=self.dtype
        )
        self.conv2 = nn.Conv(self.out_channels, (self.kernel_size,), dtype=self.dtype)
        self.drop = nn.Dropout(rate=self.p_dropout)

    def __call__(self, x, x_mask, deterministic: bool = True):
        x = self.conv1(self.paddding(x * x_mask))
        if self.activation == "gelu":
            x = x * nn.sigmoid(1.702 * x)
        else:
            x = nn.relu(x)
        x = self.drop(x, deterministic=deterministic)
        x = self.conv2(self.paddding(x * x_mask))
        return x * x_mask

    def paddding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = ((0, 0), (0, 0), (pad_l, pad_r))
        return jnp.pad(x, padding, mode="constant", constant_values=0)
