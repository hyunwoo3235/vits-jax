import numpy as np

try:
    from .monotonic_align.core import maximum_path_c
except ImportError:
    maximum_path_c = None
    print("Cython extension not found. Using pure python implementation.")


def maximum_path_each(path, value, t_y, t_x, max_neg_val=-1e9):
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            if x == y:
                v_cur = max_neg_val
            else:
                v_cur = value[y - 1, x]
            if x == 0:
                if y == 0:
                    v_prev = 0.0
                else:
                    v_prev = max_neg_val
            else:
                v_prev = value[y - 1, x - 1]
            value[y, x] += max(v_prev, v_cur)

    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
            index = index - 1


def maximum_path_py(paths, values, t_ys, t_xs):
    b, t_t, t_s = values.shape
    for i in range(b):
        maximum_path_each(paths[i], values[i], t_ys[i], t_xs[i])


def maximum_path(arg):
    """Cython optimized version.
    only require tuple of (neg_cent, mask) as input
    """
    neg_cent, mask = arg
    dtype = neg_cent.dtype
    neg_cent = neg_cent.astype(np.float32)
    path = np.zeros(neg_cent.shape, dtype=np.int32)

    t_t_max = mask.sum(1)[:, 0].astype(np.int32)
    t_s_max = mask.sum(2)[:, 0].astype(np.int32)
    if maximum_path_c is not None:
        maximum_path_c(path, neg_cent, t_t_max, t_s_max)
    else:
        maximum_path_py(path, neg_cent, t_t_max, t_s_max)

    return path.astype(dtype)
