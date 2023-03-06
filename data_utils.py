import tensorflow as tf


def getTextAudioDataset(
    tfrecords, batch_size, shuffle=True, shuffle_buffer_size=1000, compression_type=None
):
    def _parse_function(example_proto):
        feature_description = {
            "text": tf.io.FixedLenFeature([], tf.string),
            "wav": tf.io.FixedLenFeature([], tf.string),
            "spec": tf.io.FixedLenFeature([], tf.string),
            "text_length": tf.io.FixedLenFeature([], tf.int64),
            "wav_length": tf.io.FixedLenFeature([], tf.int64),
            "spec_length": tf.io.FixedLenFeature([], tf.int64),
        }

        example = tf.io.parse_single_example(example_proto, feature_description)

        example["text"] = tf.io.decode_raw(example["text"], tf.int32)
        example["wav"] = tf.io.decode_raw(example["wav"], tf.float32)
        example["spec"] = tf.io.decode_raw(example["spec"], tf.float32)

        example["spec"] = tf.reshape(example["spec"], (-1, 513))

        return example

    dataset = (
        tf.data.TFRecordDataset(tfrecords, compression_type=compression_type)
        .map(_parse_function)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def getTextAudioSpeakerDataset(
    tfrecords, batch_size, shuffle=True, shuffle_buffer_size=1000, compression_type=None
):
    def _parse_function(example_proto):
        feature_description = {
            "speaker": tf.io.FixedLenFeature([], tf.int64),
            "text": tf.io.FixedLenFeature([], tf.string),
            "wav": tf.io.FixedLenFeature([], tf.string),
            "spec": tf.io.FixedLenFeature([], tf.string),
            "text_length": tf.io.FixedLenFeature([], tf.int64),
            "wav_length": tf.io.FixedLenFeature([], tf.int64),
            "spec_length": tf.io.FixedLenFeature([], tf.int64),
        }

        example = tf.io.parse_single_example(example_proto, feature_description)

        example["speaker"] = tf.cast(example["speaker"], tf.int32)
        example["text"] = tf.io.decode_raw(example["text"], tf.int32)
        example["wav"] = tf.io.decode_raw(example["wav"], tf.float32)
        example["spec"] = tf.io.decode_raw(example["spec"], tf.float32)

        example["spec"] = tf.reshape(example["spec"], (-1, 513))

        return example

    dataset = (
        tf.data.TFRecordDataset(tfrecords, compression_type=compression_type)
        .map(_parse_function)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset
