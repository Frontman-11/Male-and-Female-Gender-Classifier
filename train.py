def train_model(model,
                X_train,
                valid_set,
                epochs,
                optimiser,
                loss,
                retrieve_weights_from=None,
                save_weights_to=None,
                use_saved_weights=True,
                with_aug_layer=True,
                callbacks=None):
    """
    Trains the model with optional loading and saving of weights, and dynamic adjustment of learning rates based on augmentation.

    Args:
        model (tf.keras.Model): The Keras model to be trained.
        X_train (tuple): Tuple containing the training data and labels. Format: (training_data, labels).
        valid_set (tuple): Tuple containing the validation data and labels. Format: (validation_data, labels).
        epochs (int): Number of epochs to train the model.
        optimiser (tf.keras.optimizers.Optimizer): Optimizer to be used for training the model.
        loss (str or tf.keras.losses.Loss): Loss function to be used for training.
        retrieve_weights_from (str, optional): Path to the weights file to load from before training. Defaults to None.
        save_weights_to (str, optional): Path to save the model weights after training. Defaults to None.
        use_saved_weights (bool, optional): Whether to load weights from `retrieve_weights_from`. Defaults to True.
        with_aug_layer (bool, optional): Whether to adjust learning rate based on augmentation. Defaults to True.
        callbacks (dict, optional): Dictionary of Keras callbacks. Keys should be callback class names, and values should be callback instances.

    Returns:
        tf.keras.callbacks.History: The training history object.
    """
    if callbacks is None:
        callbacks = {}
    cb_keys = callbacks.keys()

    # Load weights if specified
    if use_saved_weights and retrieve_weights_from:
        model.load_weights(retrieve_weights_from)
        best_val_accuracy = model.evaluate(valid_set, return_dict=True)['accuracy']
    else:
        best_val_accuracy = 0

    # Adjust callback parameters if present
    if 'EarlyStopping' in cb_keys:
        callbacks['EarlyStopping'].patience = epochs // 4
    
    if 'ModelCheckpoint' in cb_keys:
        callbacks['ModelCheckpoint'].initial_value_threshold = best_val_accuracy
        callbacks['ModelCheckpoint'].filepath = save_weights_to
    
    # Adjust learning rate based on with_aug_layer parameter
    if with_aug_layer:
        print(f'Training with with_aug_layer={with_aug_layer}')
        optimiser.learning_rate = 0.01
    else:
        print(f'Training with with_aug_layer={with_aug_layer}')
        optimiser.learning_rate = 0.001

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
    )

    return history