import jax
import jax.numpy as jnp


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def slice_segments(x, ids_str, segment_size=4):
    ret = jnp.zeros_like(x[:, :segment_size, :])
    for i in range(x.shape[0]):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret = ret.at[i].set(x[i, idx_str:idx_end, :])
    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4, rng=jax.random.PRNGKey(0)):
    b, t, d = x.shape
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size - 1
    ids_str = jax.random.uniform(rng, [b], minval=0, maxval=ids_str_max).astype(
        jnp.int32
    )
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = jnp.arange(max_length, dtype=length.dtype)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = jnp.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).astype(jnp.float32)
    path = path.view(b, t_x, t_y)
    path = path - jnp.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


def maximum_path_jax(value, mask, max_neg_val=None):
    if max_neg_val is None:
        max_neg_val = -jnp.inf
    value = value * mask

    b, t_x, t_y = value.shape
    direction = jnp.zeros(value.shape, dtype=jnp.int32)
    v = jnp.zeros((b, t_x), dtype=jnp.float32)
    x_range = jnp.arange(t_x, dtype=jnp.float32).reshape(1, -1)
    for j in range(t_y):
        v0 = jnp.pad(v, ((0, 0), (0, 1)), mode="constant", constant_values=max_neg_val)[
            :, :-1
        ]
        v1 = v
        max_mask = v1 >= v0
        v_max = jnp.where(max_mask, v1, v0)
        direction = direction.at[:, :, j].set(max_mask)

        index_mask = x_range <= j
        v = jnp.where(index_mask, v_max + value[:, :, j], max_neg_val)
    direction = jnp.where(mask, direction, 1)

    path = jnp.zeros(value.shape, dtype=jnp.float32)
    index = mask[:, :, 0].sum(1).astype(jnp.int32) - 1
    index_range = jnp.arange(b)
    for j in reversed(range(t_y)):
        path = path.at[index_range, index, j].set(1)
        index = index + direction[index_range, index, j] - 1
    return path * mask
