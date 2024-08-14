# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-08-14T17:51:14.242848Z","iopub.execute_input":"2024-08-14T17:51:14.243796Z","iopub.status.idle":"2024-08-14T17:51:14.254463Z","shell.execute_reply.started":"2024-08-14T17:51:14.243746Z","shell.execute_reply":"2024-08-14T17:51:14.252907Z"}}
import tensorflow as tf


def return_pred_and_label(dataset, count=-1):
    predictions = tf.constant([], dtype=tf.int64)
    y_true = tf.constant([], dtype=tf.int64)

    for image, label in dataset.take(count, model=None):
        y_pred = tf.cast((model.predict(image, verbose=0) >= 0.5).reshape(-1), dtype=tf.int64)
        predictions = tf.concat([predictions, y_pred], -1)
        y_true = tf.concat([y_true, label], -1)
    return {'y_true':y_true, 'y_pred':predictions}


def init_weights(model, 
                 dataset=None, 
                 retrieve_weights_from:str=None,
                 return_evaluation:bool=False):
    
    model.load_weights(retrieve_weights_from)
    
    if return_evaluation and dataset:
        evaluation = model.evaluate(dataset, return_dict=True)
        return model, evaluation 
    else:
        return model