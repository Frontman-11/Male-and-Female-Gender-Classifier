import os

# Base directories
BASE_DIR = os.path.abspath(os.path.join('..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
RAW_DIR = os.path.join(DATA_DIR, 'raw')

# Patterns for TFRecord files
def get_tfrecord_pattern(set_type, gender, data_dir=PROCESSED_DIR):
    if set_type=='train':
        return os.path.join(data_dir, f'{gender}-tfrecord-dataset', f'{set_type}_{gender}.tfrecord-*-of-*')
    return os.path.join(data_dir, f'{set_type}-tfrecord-dataset', f'{set_type}_{gender}.tfrecord-*-of-*')

# Paths to specific TFRecord patterns
TRAIN_MALE_PATTERN = get_tfrecord_pattern('train', 'male')
TRAIN_FEMALE_PATTERN = get_tfrecord_pattern('train', 'female')

VALID_MALE_PATTERN = get_tfrecord_pattern('valid', 'male')
VALID_FEMALE_PATTERN = get_tfrecord_pattern('valid', 'female')

TEST_MALE_PATTERN = get_tfrecord_pattern('test', 'male')
TEST_FEMALE_PATTERN = get_tfrecord_pattern('test', 'female')