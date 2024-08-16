import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def load_evaluation_data(dataset_path, batch_size=32):
    # Implement data loading and preprocessing here
    return dataset


def evaluate_model(model, dataset):
    result = model.evaluate(dataset, return_dict=True)
    print(f'{result}')
    return result

    
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


def generate_classification_report(model, dataset):
    print(classification_report(return_pred_and_label(model, dataset).values()))
    
    
def plot_confusion_matrix(model, dataset, figsize=(10, 7)):
    y_true, y_pred = return_pred_and_label(model, dataset).values()
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()