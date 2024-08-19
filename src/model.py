import tensorflow as tf

class AugmentationLayer(tf.keras.layers.Layer):
    def __init__(self, augmentation_model, **kwargs):
        super(AugmentationLayer, self).__init__(**kwargs)
        self.augmentation_model = augmentation_model
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "augmentation_model": self.augmentation_model
        })
        return config
    
    def build(self, input_shape):
        pass
        
    def call(self, x, training=False):
        if training:
            return self.augmentation_model(x, training=training)
        return x
        
def create_augmentation_model(name=None):
    augmentation_model = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.01),
        tf.keras.layers.RandomTranslation(height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05)),
        tf.keras.layers.RandomBrightness(0.5),
        tf.keras.layers.Lambda(lambda x: x/255.0)
    ], name=name)
    
    return AugmentationLayer(augmentation_model, name=name)

def create_model(
    filters=32,
    kernel_size=(3,3),
    kernel_initializer='he_normal', 
    use_bias=False,
    padding='same',
    input_shape=(218, 178, 3)):
    
    """
    Constructs a CNN model with augmentation and multiple convolutional layers.

    Args:
        filters (int, optional): Number of filters in the first convolutional layer. Default is 32.
        kernel_size (tuple, optional): Size of the convolutional kernels. Default is (3, 3).
        kernel_initializer (str, optional): Kernel initializer. Default is 'he_normal'.
        use_bias (bool, optional): Whether to use bias in convolutional and dense layers. Default is False.
        padding (str, optional): Padding for convolutional layers. Default is 'same'.
        input_shape (tuple, optional): Shape of the input images. Default is (218, 178, 3).

    Returns:
        tf.keras.Model: The constructed Keras model.
    """
    
    model = tf.keras.Sequential(
    layers=[
        tf.keras.layers.Input(shape=input_shape, name='human_face'),
        create_augmentation_model(name='augmentation_layer'),
        tf.keras.layers.Conv2D(filters=filters, kernel_size=(5,5), padding=padding, kernel_initializer=kernel_initializer, use_bias=use_bias, name='cov2d_1'),
        tf.keras.layers.BatchNormalization(name='batchnorm_1'),
        tf.keras.layers.Activation(tf.keras.activations.swish,  name='swish_activation_1'),
        tf.keras.layers.MaxPool2D(name='maxpool2d_1'),

        tf.keras.layers.Conv2D(filters=filters*2, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer, use_bias=use_bias, name='cov2d_2'),
        tf.keras.layers.BatchNormalization(name='batchnorm_2'),
        tf.keras.layers.Activation(tf.keras.activations.swish, name='swish_activation_2'),
        tf.keras.layers.MaxPool2D(name='maxpool2d_2'),

        tf.keras.layers.Conv2D(filters=filters*3, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer, use_bias=use_bias, name='cov2d_3'),
        tf.keras.layers.BatchNormalization(name='batchnorm_3'),
        tf.keras.layers.Activation(tf.keras.activations.swish, name='swish_activation_3'),
        tf.keras.layers.MaxPool2D(name='maxpool2d_3'),

        tf.keras.layers.Conv2D(filters=filters*4, padding=padding, kernel_size=kernel_size, kernel_initializer=kernel_initializer, use_bias=use_bias, name='cov2d_4'),
        tf.keras.layers.BatchNormalization(name='batchnorm_4'),
        tf.keras.layers.Activation(tf.keras.activations.swish, name='swish_activation_4'),
        tf.keras.layers.MaxPool2D(name='maxpool2d_4'),

        tf.keras.layers.Flatten(name='flatten_layer_1'),
        tf.keras.layers.Dropout(rate=0.3, name='dropout_layer_1'),
        tf.keras.layers.Dense(units=512, use_bias=use_bias, name='dense_layer_1'),
        tf.keras.layers.BatchNormalization(name='batchnorm_5'),
        tf.keras.layers.Activation(tf.keras.activations.swish, name='swish_activation_5'),
        tf.keras.layers.Dense(1, activation='sigmoid', name='sigmoid_output')
    ], name='Gender_clf_model')
    return model


def init_weights(model, 
                 retrieve_weights_from: str,
                 dataset=None,
                 params=None,
                 return_evaluation: bool = False):
    """
    Load weights into a model and optionally evaluate it on a dataset.

    Args:
        model (tf.keras.Model): Model to load weights into.
        retrieve_weights_from (str): Path to weight file. Defaults to None.
        dataset (tf.data.Dataset, optional): Dataset for evaluation. Defaults to None.
        params (dict, optional): Parameters for model.compile(). Defaults to None.
        return_evaluation (bool, optional): Whether to return evaluation results. Defaults to False.

    Returns:
        tuple: Model and optionally evaluation results.
    """
    model.load_weights(retrieve_weights_from)
    
    if return_evaluation and dataset and params:
        model.compile(**params)
        evaluation = model.evaluate(dataset, return_dict=True)
        return model, evaluation 
    else:
        return model, None
