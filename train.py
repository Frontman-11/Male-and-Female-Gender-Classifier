def train_model(model,
                X_train,
                valid_set,
                epochs,
                optimiser,
                loss,
                callbacks=None,
                retrieve_weights_from=None,
                save_weights_to=None,
                with_aug_layer=True,
                aug_layer_name=None,
                insert_dropout=None,
                save_model_to=None,
                **kwargs
               ):
    """
    Trains the model with optional loading and saving of weights, dynamic adjustment of the model based on augmentation layers, 
    and the possibility to insert dropout layers. Optionally, the trained model can be saved.

    Args:
        model (tf.keras.Model): The Keras model to be trained.
        X_train (tuple): Tuple containing the training data and labels. Format: (training_data, labels).
        valid_set (tuple): Tuple containing the validation data and labels. Format: (validation_data, labels).
        epochs (int): Number of epochs to train the model.
        optimiser (tf.keras.optimizers.Optimizer): Optimizer to be used for training the model.
        loss (str or tf.keras.losses.Loss): Loss function to be used for training.
        callbacks (dict, optional): Dictionary of Keras callbacks. Keys should be callback class names, and values should be callback instances.
        retrieve_weights_from (str, optional): Path to the weights file to load from before training. Defaults to None.
        save_weights_to (str, optional): Path to save the model weights after training. Defaults to None.
        with_aug_layer (bool, optional): Whether to adjust model based on the presence of an augmentation layer. Defaults to True.
        aug_layer_name (str, optional): Name of the augmentation layer to identify and remove if `with_aug_layer` is False. Defaults to None.
        insert_dropout (tf.keras.layers.Layer, optional): Dropout layer to be inserted before the final layer if augmentation is removed. Defaults to None.
        save_model_to (str, optional): Path to save the entire model after training. Defaults to None.
        kwargs: passed to model fit method.

    Returns:
        tf.keras.callbacks.History: The training history object.
    """
    if callbacks is None:
        callbacks = {}
    cb_keys = callbacks.keys()

    # Load weights if specified
    if retrieve_weights_from:
        model.load_weights(retrieve_weights_from)
        
        model.compile(
        loss=loss,
        optimizer=optimiser,
        metrics=['accuracy']
        )
        
        best_val_accuracy = model.evaluate(valid_set, return_dict=True)['accuracy']
        print(f'Model loaded weights with best_val_accuracy: {best_val_accuracy}')
        
    else:
        best_val_accuracy = 0

    # Adjust callback parameters if present
    if 'ModelCheckpoint' in cb_keys:
        callbacks['ModelCheckpoint'].initial_value_threshold = best_val_accuracy
        callbacks['ModelCheckpoint'].filepath = save_weights_to
    
    # Adjust learning rate based on with_aug_layer parameter
    print(f'Training with with_aug_layer={with_aug_layer}')
    
    if not with_aug_layer:
        if model.layers[0].name == 'augmentation_layer' or model.layers[0].name == aug_layer_name:
            layers = model.layers[1:]
            if insert_dropout:
                index = layers.index(layers[-1])
                layers.insert(index, insert_dropout)
            model = tf.keras.Sequential(layers=layers, name='final_gender_clf_model')

    model.compile(
        loss=loss,
        optimizer=optimiser,
        metrics=['accuracy']
    )

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

    return history