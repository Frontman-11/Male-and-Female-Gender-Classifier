def train_model(model,
                valid_set,
                epochs,
                optimiser,
                loss,
                with_aug_layer=True,
                callbacks=None
               ):
    
    if with_aug_layer:
        print(f'Training with with_aug_layer={with_aug_layer}')
        model.compile(
            loss=loss,
            optimizer=optimiser,
            metrics=['accuracy']
        )
        
        history = model.fit(
            X_train,
            epochs=epochs,
            verbose='auto',
            callbacks=callbacks,
            validation_data=valid_set,
            shuffle=True,
            validation_freq=1,
        )
        
    else:
        print(f'Training with with_aug_layer={with_aug_layer}')
        model.compile(
            loss=loss,
            optimizer=optimiser,
            metrics=['accuracy']
        )
        
        history = model.fit(
            X_train,
            epochs=epochs,
            verbose='auto',
            callbacks=callbacks,
            validation_data=valid_set,
            shuffle=True,
            validation_freq=1,
        )
        
    return history