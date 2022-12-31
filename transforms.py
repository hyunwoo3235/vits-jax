import jax
import jax.numpy as jnp

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def piecewise_rational_quadratic_transform(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails=None,
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    if tails is None:
        spline_fn = rational_quadratic_spline
        spline_kwargs = {}
    else:
        spline_fn = unconstrained_rational_quadratic_spline
        spline_kwargs = {"tails": tails, "tail_bound": tail_bound}

    outputs, logabsdet = spline_fn(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
        **spline_kwargs
    )
    return outputs, logabsdet


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return jnp.sum(inputs[..., None] >= bin_locations, axis=-1) - 1


def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = jnp.zeros_like(inputs)
    logabsdet = jnp.zeros_like(inputs)

    if tails == "linear":
        unnormalized_derivatives = jnp.pad(
            unnormalized_derivatives,
            ((0, 0), (0, 0), (0, 0), (1, 1)),
            mode="constant",
            constant_values=0.0,
        )
        constant = jnp.log(jnp.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise NotImplementedError

    (
        outputs[inside_interval_mask],
        logabsdet[inside_interval_mask],
    ) = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    return outputs, logabsdet


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    if jnp.min(inputs) < left or jnp.max(inputs) > right:
        raise ValueError("Input to a transform is not within its domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = jax.nn.softmax(unnormalized_widths, axis=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cum_widths = jnp.cumsum(widths, axis=-1)
    cum_widths = jnp.pad(
        cum_widths, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0.0
    )
    cum_widths = (right - left) * cum_widths + left
    cum_widths[..., 0] = left
    cum_widths[..., -1] = right
    widths = cum_widths[..., 1:] - cum_widths[..., :-1]

    derivatives = jax.nn.softplus(unnormalized_derivatives) + min_derivative

    heights = jax.nn.softmax(unnormalized_heights, axis=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cum_heights = jnp.cumsum(heights, axis=-1)
    cum_heights = jnp.pad(
        cum_heights, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0.0
    )
    cum_heights = (top - bottom) * cum_heights + bottom
    cum_heights[..., 0] = bottom
    cum_heights[..., -1] = top
    heights = cum_heights[..., 1:] - cum_heights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cum_heights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cum_widths, inputs)[..., None]

    input_cumwidths = jnp.take_along_axis(cum_widths, bin_idx, axis=-1)[..., 0]
    input_bin_widths = jnp.take_along_axis(widths, bin_idx, axis=-1)[..., 0]

    input_cumheights = jnp.take_along_axis(cum_heights, bin_idx, axis=-1)[..., 0]
    delta = heights / widths
    input_delta = jnp.take_along_axis(delta, bin_idx, axis=-1)[..., 0]

    input_derivatives = jnp.take_along_axis(derivatives, bin_idx, axis=-1)[..., 0]
    input_derivatives_plus_one = jnp.take_along_axis(
        derivatives[..., 1:], bin_idx, axis=-1
    )[..., 0]

    input_heights = jnp.take_along_axis(heights, bin_idx, axis=-1)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b**2 - 4 * a * c

        root = (2 * c) / (-b - jnp.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = jnp.power(input_delta, 2) * (
            input_derivatives_plus_one * jnp.power(root, 2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * jnp.power(1 - root, 2)
        )
        logabsdet = jnp.log(derivative_numerator) - 2 * jnp.log(denominator)

        return outputs, -logabsdet

    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * jnp.power(theta, 2)
            + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = jnp.power(input_delta, 2) * (
            input_derivatives_plus_one * jnp.power(theta, 2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * jnp.power(1 - theta, 2)
        )
        logabsdet = jnp.log(derivative_numerator) - 2 * jnp.log(denominator)

        return outputs, logabsdet
