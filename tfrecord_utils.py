import tensorflow as tf
import cv2
import os
from contextlib import ExitStack
from tensorflow.train import Feature, Features, Example
from tensorflow.train import BytesList, Int64List
import concurrent.futures
import threading 
from tqdm import tqdm


def create_example(image, label):
    """
    Creates a TensorFlow Example protobuf containing an image and its corresponding label.

    Args:
        image (numpy.ndarray): The image array, usually loaded via OpenCV.
        label (int): The integer label associated with the image.

    Returns:
        tensorflow.train.Example: A TensorFlow Example containing the serialized image and label.
    """
    serialized_image = tf.io.serialize_tensor(image)

    return Example(
        features=Features(
            feature={
                'images': Feature(bytes_list=BytesList(value=[serialized_image.numpy()])),
                'label': Feature(int64_list=Int64List(value=[label]))
            }
        )
    )



def helper_func(parent_dir, filename, label):
    """
    Loads an image, processes it, and creates a TensorFlow Example protobuf.

    Args:
        parent_dir (str): The directory path containing the image file.
        filename (str): The name of the image file.
        label (int): The label associated with the image.

    Returns:
        tensorflow.train.Example: A TensorFlow Example containing the serialized image and label.
    """
    filepath = os.path.join(parent_dir, filename)
    img = cv2.imread(filepath, flags=1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return create_example(img, label)



def write_tfrecord(parent_dir, idx, filename, n_shards, writers, label, writer_lock):
    example = helper_func(parent_dir, filename, label)
    shard = idx % n_shards  # Determine the shard index
    with writer_lock[shard]:  # Ensure thread-safe writing
        writers[shard].write(example.SerializeToString())



def write_as_tfrecord(parent_dir, tfrecord_filename, label, n_shards=10, max_workers=3000):
    """
    Converts a directory of image files into a set of TFRecord files, using concurrent processing.

    Args:
        parent_dir (str): The directory containing the image files.
        tfrecord_filename (str): The base filename for the TFRecord files.
        label (int): The label associated with the images.
        n_shards (int, optional): The number of TFRecord files to create (default is 10).
        max_workers (int, optional): The maximum number of threads to use (default is 3000).

    Returns:
        list: A list of paths to the created TFRecord files.
    """
    filenames = os.listdir(parent_dir)
    pad = len(str(n_shards))  # Calculate padding length for filenames
    paths = [f'{tfrecord_filename}.tfrecord-{index+1:0{pad}d}-of-{n_shards:0{pad}d}' for index in range(n_shards)]
    writer_lock = [threading.Lock() for _ in range(n_shards)] 
    
    with ExitStack() as stack:
        writers = [stack.enter_context(tf.io.TFRecordWriter(path)) for path in paths]

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(write_tfrecord, parent_dir, idx, filename, n_shards, writers, label, writer_lock)
                           for idx, filename in enumerate(filenames)]

                for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Writing TFRecords"):
                    pass

                for future in concurrent.futures.as_completed(futures):
                    future.result()

        except tf.errors.ResourceExhaustedError as e:
            print('\nProgram stopped due to limited storage space')
            return e
        
    return paths