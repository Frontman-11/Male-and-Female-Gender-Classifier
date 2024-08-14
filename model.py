# %% [code]
# %% [code] {"execution":{"iopub.status.busy":"2024-08-10T21:34:16.015862Z","iopub.execute_input":"2024-08-10T21:34:16.016768Z","iopub.status.idle":"2024-08-10T21:34:30.581333Z","shell.execute_reply.started":"2024-08-10T21:34:16.016725Z","shell.execute_reply":"2024-08-10T21:34:30.580151Z"},"jupyter":{"outputs_hidden":false}}
import tensorflow as tf

# %% [code] {"execution":{"iopub.status.busy":"2024-08-10T21:36:00.302626Z","iopub.execute_input":"2024-08-10T21:36:00.303635Z","iopub.status.idle":"2024-08-10T21:36:00.320744Z","shell.execute_reply.started":"2024-08-10T21:36:00.303595Z","shell.execute_reply":"2024-08-10T21:36:00.319618Z"},"jupyter":{"outputs_hidden":false}}
class AugmentationLayer(tf.keras.layers.Layer):
    def __init__(self, augmentation_model, **kwargs):
        super(AugmentationLayer, self).__init__(**kwargs)
        self.augmentation_model=augmentation_model
    
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
        
        
    
def create_augmentation_model():
    augmentation_model = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.01),
        tf.keras.layers.RandomTranslation(height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05)),
        tf.keras.layers.RandomBrightness(0.5),
        tf.keras.layers.Lambda(lambda x: x/255.0)
    ])
    return AugmentationLayer(augmentation_model)

 

def create_model(
    filters = 32,
    kernel_size= (3,3),
    kernel_initializer = 'he_normal', 
    use_bias = False,
    padding = 'same',
    input_shape = (218, 178, 3)):
    
    model = tf.keras.Sequential(
    layers= [
        tf.keras.layers.Input(shape=input_shape, name='human_face'),
        create_augmentation_model(),
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
    ])
    return model