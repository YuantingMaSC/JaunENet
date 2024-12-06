import tensorflow as tf
from configuration import metasize


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto.
    return tf.io.parse_single_example(example_proto, {
        'label': tf.io.FixedLenFeature([], tf.dtypes.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.dtypes.string),
        'meta': tf.io.FixedLenFeature([metasize], tf.dtypes.float32)
    })


def get_parsed_dataset(tfrecord_name):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
    parsed_dataset = raw_dataset.map(_parse_image_function)

    return parsed_dataset