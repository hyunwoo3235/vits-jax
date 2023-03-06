import argparse

import librosa
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import read
from tqdm.auto import tqdm

import commons
from mel_processing import spectrogram_jax
from text.process_ko import cleaned_ko_text_to_sequence


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_name", default="kss")
    parser.add_argument(
        "--filelists", default="filelists/kss_audio_text_train_filelist.txt"
    )
    parser.add_argument("--add_blank", default=True, type=bool)
    parser.add_argument("--max_wav_length", default=220500, type=int)
    parser.add_argument("--min_wav_length", default=0, type=int)
    parser.add_argument("--max_text_length", default=160, type=int)
    parser.add_argument("--min_text_length", default=0, type=int)

    args = parser.parse_args()

    with open(args.filelists, "r") as f:
        filepaths_and_text = [line.strip().split("|") for line in f]

    max_spec_length = spectrogram_jax(
        np.zeros((1, args.max_wav_length, 1)),
        1024,
        256,
        1024,
    ).shape[1]

    with tf.io.TFRecordWriter(
        f"{args.out_name}.tfrecord",
        options=tf.io.TFRecordOptions(compression_type="GZIP"),
    ) as writer:
        for i, data in tqdm(
            enumerate(filepaths_and_text), total=len(filepaths_and_text)
        ):
            filepath, text = data[0], data[1]

            text = cleaned_ko_text_to_sequence(text)
            if args.add_blank:
                text = commons.intersperse(text, 0)
            if not args.min_text_length <= len(text) <= args.max_text_length:
                continue

            sr, wav = read(filepath)
            if not args.min_wav_length <= len(wav) <= args.max_wav_length:
                continue

            wav = librosa.to_mono(wav.T.astype(np.float32))
            wav = librosa.resample(wav, orig_sr=sr, target_sr=22050)
            wav = np.divide(wav, 32768.0)

            spec = spectrogram_jax(
                wav.reshape(1, -1, 1),
                1024,
                256,
                1024,
            )

            text = np.pad(
                text, (0, args.max_text_length - len(text)), "constant"
            ).astype(np.int32)
            wav = np.pad(wav, (0, args.max_wav_length - len(wav)), "constant").astype(
                np.float32
            )
            spec = np.pad(
                spec, ((0, 0), (0, max_spec_length - spec.shape[1]), (0, 0)), "constant"
            ).astype(np.float32)

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "text": _bytes_feature(text.tobytes()),
                        "wav": _bytes_feature(wav.tobytes()),
                        "spec": _bytes_feature(spec.tobytes()),
                    }
                )
            )
            writer.write(example.SerializeToString())
