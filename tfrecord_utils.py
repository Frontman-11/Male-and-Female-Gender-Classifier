import tensorflow as tf
import cv2
import os
from contextlib import ExitStack
from tensorflow.train import Feature, Features, Example
from tensorflow.train import BytesList, Int64List
import concurrent.futures
import threading 

def create_example(image, label):
    '''Returns example protobuf'''
    serialised_image = tf.io.serialize_tensor(image)

    return Example(
                features = Features(
                    feature = {
                        'images': Feature(bytes_list=BytesList(value=[serialised_image.numpy()])),
                        'label': Feature(int64_list=Int64List(value=[label]))
                    }
                )
            )


def helper_func(parent_dir, filename, label):      
    filepath = os.path.join(parent_dir, filename)
    img = cv2.imread(filepath, flags=1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return create_example(img, label)



def write_tfrecord(parent_dir, idx, filename, file_count, n_shards, writers, label):   
    example = helper_func(parent_dir, filename, label, writer_lock)
    # Determine shard
    shard = idx % n_shards
    with writer_lock[shard]:
        writers[shard].write(example.SerializeToString())
#         file_count[0] += 1
#         print(f'\rProgress:{file_count[0]} of {len(os.listdir(parent_dir))}', end='', flush=True)        
        
#         if file_count[0]%10 == 0:
#         with threading.Lock():
#     print(f'\rProgress:{file_count[0]} of {len(os.listdir(parent_dir))} \t{(file_count[0]*100/len(os.listdir(parent_dir))):.2f}% complete', end='', flush=True)



def write_as_tfrecord(parent_dir, tfrecord_filename, label, n_shards=10, max_workers=3000):
    file_count = [0]
    count = [0]
    
    filenames = os.listdir(parent_dir)
    pad = len(str(n_shards))
    paths = [f'{tfrecord_filename}.tfrecord-{index+1:0{pad}d}-of-{n_shards:0{pad}d}' for index in range(n_shards)]        
    writer_lock = [threading.Lock() for _ in range (20)]

    with ExitStack() as stack:
        writers = [stack.enter_context(tf.io.TFRecordWriter(path)) for path in paths]
        try:
            if __name__ == '__main__':        
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(write_tfrecord, parent_dir, idx, filename, file_count, n_shards,writers,
                                               label, writer_lock) for idx, filename in enumerate(filenames)]
                    for future in concurrent.futures.as_completed(futures):
                        future.result()                                                                                                                                                                                                                                                                                                                                                                                                                                                          
            else:
                raise ValueError('Name not equal to __main__')
                
        except ResourceExhaustedError as e:
            print('\nProgram stopped due to limited storage space')
            return e
    print('done writing')
    return paths