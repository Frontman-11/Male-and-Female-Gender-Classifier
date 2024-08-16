import tensorflow as tf
from tensorflow.train import Feature, Features, Example
from tensorflow.train import BytesList, Int64List


def parse_example(serialised_example):
    feature = {
        'images': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        'label': tf.io.FixedLenFeature(shape=[], dtype=tf.int64)
    }
    return tf.io.parse_single_example(serialised_example, feature)


def normalise_image(image_batch):
    
    """
    Normalizes a batch of images by scaling the pixel values to the range [0, 1].

    Args:
        image_batch (tf.Tensor): A batch of images with pixel values in the range [0, 255].

    Returns:
        tf.Tensor: A batch of images with pixel values scaled to the range [0, 1].
    """
        
    image_batch = tf.cast(image_batch, tf.float32)/255.
    return image_batch

    
def gender_dataset(filepaths, 
                   repeat=None,
                   normalise=True,
                   augment_image=None,
                   shuffle_buffer_size=None,
                   cache=None, 
                   batch_size=32,
                   image_shape=(218, 178, 3),
                   n_reads=tf.data.AUTOTUNE,
                   prefetch=tf.data.AUTOTUNE,
                  ):
    
    """
    Creates a TensorFlow Dataset for gender classification from a list of TFRecord file paths.

    Args:
        filepaths (list of str): List of paths to the TFRecord files.
        repeat (int, optional): Number of times to repeat the dataset. Defaults to None (no repetition).
        normalise (bool, optional): Whether to normalize the images to [0, 1] range. Defaults to True.
        augment (function, optional): A function to apply data augmentation to images. Defaults to None.
        shuffle_buffer_size (int, optional): Size of the buffer for shuffling. Defaults to None (no shuffling).
        cache (str or bool, optional): If True, cache the dataset in memory; if a string, cache to the specified file. Defaults to None.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        image_shape (tuple, optional): Expected shape of the images (height, width, channels). Defaults to (218, 178, 3).
        n_reads (int, optional): Number of parallel reads from the TFRecord files. Defaults to tf.data.AUTOTUNE.
        prefetch (int, optional): Number of batches to prefetch for performance. Defaults to tf.data.AUTOTUNE.

    Returns:
        tf.data.Dataset: A TensorFlow Dataset object ready for training or evaluation.
    """
        
    dataset = tf.data.TFRecordDataset(filepaths, num_parallel_reads=n_reads)
    dataset = dataset.map(lambda serialised_example: parse_example(serialised_example), num_parallel_calls=n_reads)
    dataset = dataset.map(lambda example: (tf.io.parse_tensor(example['images'], out_type=tf.uint8), example['label']), num_parallel_calls=n_reads)
    dataset = dataset.map(lambda image, label: (tf.ensure_shape(image, shape=image_shape), label), num_parallel_calls=n_reads)
    
    if repeat:
        dataset = dataset.repeat(repeat)
        
    if shuffle_buffer_size:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        
    if augment_image:
        dataset = dataset.map(augment_image, num_parallel_calls=n_reads)
        
    if cache:
        dataset = dataset.cache(cache)
        
    if normalise:
        dataset = dataset.map(lambda image, label: (normalise_image(image), label), num_parallel_calls=n_reads)

    dataset = dataset.batch(batch_size=batch_size)
    return dataset.prefetch(prefetch)