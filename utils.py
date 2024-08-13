# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-08-13T00:42:25.645337Z","iopub.execute_input":"2024-08-13T00:42:25.645766Z","iopub.status.idle":"2024-08-13T00:42:25.653973Z","shell.execute_reply.started":"2024-08-13T00:42:25.645735Z","shell.execute_reply":"2024-08-13T00:42:25.652425Z"}}
import tensorflow as tf


def return_pred_and_label(dataset, count=-1):
    predictions = tf.constant([], dtype=tf.int64)
    y_true = tf.constant([], dtype=tf.int64)

    for image, label in dataset.take(count, model=None):
        y_pred = tf.cast((model.predict(image, verbose=0) >= 0.5).reshape(-1), dtype=tf.int64)
        predictions = tf.concat([predictions, y_pred], -1)
        y_true = tf.concat([y_true, label], -1)
    return {'y_true':y_true, 'y_pred':predictions}
