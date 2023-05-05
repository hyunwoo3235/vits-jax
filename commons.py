import jax
import jax.numpy as jnp


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def slice_segments(x, ids_str, segment_size=4):
    ret = jnp.zeros_like(x[:, :segment_size, :])
    for i in range(x.shape[0]):
        idx = jnp.add(jnp.arange(segment_size), ids_str[i])
        ret = ret.at[i].set(jnp.take(x[i], idx, axis=0))
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


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = jnp.arange(max_length, dtype=length.dtype)
    return jnp.expand_dims(x, 0) < jnp.expand_dims(length, 1)


def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = jnp.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).astype(jnp.float32)
    path = jnp.reshape(path, (b, t_x, t_y))
    path = path - jnp.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = jnp.expand_dims(path, 1).transpose(0, 1, 3, 2) * mask
    return path
