import tensorflow as tf


def train_model(model,
                X_train,
                valid_set,
                epochs,
                params=None,
                callbacks=None,
                with_aug_layer=True,
                aug_layer_name=None,
                insert_dropout=None,
                save_model_to=None,
                show_model_summary=False,
                **kwargs):
    """
    Trains the given Keras model with options for dynamic adjustments, augmentation layer management, and dropout insertion.
    Optionally saves the trained model.

    Args:
        model (tf.keras.Model): The Keras model to be trained.
        X_train (tuple): Tuple containing training data and labels. Format: (training_data, labels).
        valid_set (tuple): Tuple containing validation data and labels. Format: (validation_data, labels).
        epochs (int): Number of epochs to train the model.
        params (dict, optional): Parameters for model compilation, e.g., {'optimizer': optimizer, 'loss': loss_function}. Defaults to None.
        callbacks (dict, optional): Dictionary of Keras callbacks. Keys should be callback class names, values should be callback instances. Defaults to None.
        with_aug_layer (bool, optional): Whether to include the augmentation layer in the model. Defaults to True.
        aug_layer_name (str, optional): Name of the augmentation layer to identify and remove if `with_aug_layer` is False. Defaults to None.
        insert_dropout (tf.keras.layers.Layer, optional): Dropout layer to be inserted before the final layer if the augmentation layer is removed. Defaults to None.
        save_model_to (str, optional): Path to save the model after training. Defaults to None.
        show_model_summary (bool, optional): Whether to print the model summary. Defaults to False.
        **kwargs: Additional arguments passed to the `model.fit` method.

    Returns:
        tf.keras.callbacks.History: The training history object.
    """
    if callbacks is None:
        callbacks = {}

    # Display augmentation layer status
    print(f'Training with with_aug_layer={with_aug_layer}')
    
    # Adjust model if not using augmentation layer
    if not with_aug_layer:
        if model.layers[0].name == 'augmentation_layer' or model.layers[0].name == aug_layer_name:
            layers = model.layers[1:]
            if insert_dropout:
                # Insert dropout layer before the final layer
                index = len(layers) - 1
                layers.insert(index, insert_dropout)
            model = tf.keras.Sequential(layers=layers, name='final_gender_clf_model')
            
    if show_model_summary:
        # Print model summary
        model.summary()

    if params:
        # Compile the model with provided parameters
        model.compile(**params)

    # Train the model
    history = model.fit(
        X_train,
        epochs=epochs,
        verbose='auto',
        callbacks=list(callbacks.values()),
        validation_data=valid_set,
        shuffle=True,
        validation_freq=1,
        **kwargs
    )
    
    if save_model_to:
        try:
            model.save(save_model_to)
        except Exception as e:
            print(f"Error saving model: {e}")

    return model, history
