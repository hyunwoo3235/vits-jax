import jax.numpy as jnp
from audax.core.functional import melscale_fbanks, spectrogram, apply_melscale

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_jax(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return jnp.log(jnp.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression_jax(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return jnp.exp(x) / C


def spectral_normalize_jax(magnitudes):
    output = dynamic_range_compression_jax(magnitudes)
    return output


def spectral_de_normalize_jax(magnitudes):
    output = dynamic_range_decompression_jax(magnitudes)
    return output


mel_basis = None
hann_window = None


def mel_setup(n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax):
    global hann_window, mel_basis
    if hann_window is None:
        hann_window = jnp.hanning(win_size)
    if mel_basis is None:
        mel_basis = melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            n_mels=num_mels,
            sample_rate=sampling_rate,
            f_min=fmin,
            f_max=fmax,
            norm="slaney",
            mel_scale="slaney",
        )


def spectrogram_jax(y, n_fft, hop_size, win_size, center=False):
    global hann_window
    if hann_window is None:
        hann_window = jnp.hanning(win_size)

    y = jnp.pad(
        y,
        ((0, 0), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), (0, 0)),
        mode="reflect",
    )

    spec = spectrogram(
        y,
        pad=0,
        window=hann_window,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        power=1.0,
        normalized=False,
        center=center,
        onesided=True,
    )
    return spec


def spec_to_mel_jax(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    if mel_basis is None:
        mel_basis = melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            n_mels=num_mels,
            sample_rate=sampling_rate,
            f_min=fmin,
            f_max=fmax,
            norm="slaney",
            mel_scale="slaney",
        )
    mel = apply_melscale(spec, mel_basis)
    mel = spectral_normalize_jax(mel)

    return mel


def mel_spectrogram_jax(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    global hann_window, mel_basis
    if hann_window is None:
        hann_window = jnp.hanning(win_size)
    if mel_basis is None:
        mel_basis = melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            n_mels=num_mels,
            sample_rate=sampling_rate,
            f_min=fmin,
            f_max=fmax,
            norm="slaney",
            mel_scale="slaney",
        )

    y = jnp.pad(
        y,
        ((0, 0), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), (0, 0)),
        mode="reflect",
    )

    spec = spectrogram(
        y,
        pad=0,
        window=hann_window,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        power=1.0,
        normalized=False,
        center=center,
        onesided=True,
    )

    mel = apply_melscale(spec, mel_basis)
    mel = spectral_normalize_jax(mel)

    return mel
