# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-08-13T01:10:35.041848Z","iopub.execute_input":"2024-08-13T01:10:35.042283Z","iopub.status.idle":"2024-08-13T01:10:35.047993Z","shell.execute_reply.started":"2024-08-13T01:10:35.042252Z","shell.execute_reply":"2024-08-13T01:10:35.046701Z"}}
import tensorflow as tf
from tensorflow.train import Feature, Features, Example
from tensorflow.train import BytesList, Int64List

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-08-13T00:52:12.341284Z","iopub.execute_input":"2024-08-13T00:52:12.342457Z","iopub.status.idle":"2024-08-13T00:52:12.353347Z","shell.execute_reply.started":"2024-08-13T00:52:12.342417Z","shell.execute_reply":"2024-08-13T00:52:12.352233Z"}}
def parse_example(serialised_example):
    feature = {
        'images': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        'label': tf.io.FixedLenFeature(shape=[], dtype=tf.int64)
    }
    return tf.io.parse_single_example(serialised_example, feature)


def normalise_image(image_batch):
    image_batch = tf.cast(image_batch, tf.float32)/255.
    return image_batch

    
def gender_dataset(filepaths, 
                   repeat=None,
                   normalise=True,
                   augment=None,
                   shuffle_buffer_size=None,
                   cache=None, 
                   batch_size=32,
                   image_shape=(218, 178, 3),
                   n_reads=tf.data.AUTOTUNE,
                   prefetch=tf.data.AUTOTUNE,
                  ):
        
    dataset = tf.data.TFRecordDataset(filepaths, num_parallel_reads=n_reads)
    dataset = dataset.map(lambda serialised_example: parse_example(serialised_example), num_parallel_calls=n_reads)
    dataset = dataset.map(lambda example: (tf.io.parse_tensor(example['images'], out_type=tf.uint8), example['label']), num_parallel_calls=n_reads)
    dataset = dataset.map(lambda image, label: (tf.ensure_shape(image, shape=image_shape), label), num_parallel_calls=n_reads)
    
    if repeat:
        dataset = dataset.repeat(repeat)
        
    if shuffle_buffer_size:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        
    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=n_reads)
        
    if cache:
        dataset = dataset.cache()
        
    if normalise:
        dataset = dataset.map(lambda image, label: (tf.cast(image, dtype=tf.float32)/255., label), num_parallel_calls=n_reads)
    dataset = dataset.batch(batch_size=batch_size)
    
    return dataset.prefetch(prefetch)