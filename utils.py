import tensorflow as tf


def return_pred_and_label(dataset, model, count=-1):
    """
    Generate predictions and true labels from a dataset using a model.

    Args:
        dataset (tf.data.Dataset): Dataset containing images and labels.
        model (tf.keras.Model): Trained model for predictions. Defaults to None.
        count (int, optional): Number of batches to process. Defaults to -1 (process all).

    Returns:
        dict: Dictionary with 'y_true' and 'y_pred' tensors.
    """
    predictions = tf.constant([], dtype=tf.int64)
    y_true = tf.constant([], dtype=tf.int64)

    for image, label in dataset.take(count):
        y_pred = tf.cast((model.predict(image, verbose=0) >= 0.5).reshape(-1), dtype=tf.int64)
        predictions = tf.concat([predictions, y_pred], -1)
        y_true = tf.concat([y_true, label], -1)
    return {'y_true': y_true, 'y_pred': predictions}


def stack_filepaths(paths: list, max_length=None):
    """
    Stack file paths into a single list, repeating entries to match the maximum length.

    Args:
        paths (list of list of str): List of lists containing file paths.
        max_length (int, optional): Maximum length to stack file paths. Defaults to None.

    Returns:
        list of str: Stacked file paths.
    """
    length = max_length or max(len(path) for path in paths)
    
    filepaths = []
    for i in range(length):
        for path in paths:
            try:
                filepaths.extend([path[i]])
            except IndexError:
                continue
    return filepaths


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
