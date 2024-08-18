# %% [code]
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def _return_pred_and_label(model, dataset, count):
    """
    Generate predictions and true labels from a dataset using a model.

    Args:
        model (tf.keras.Model): Trained model used for predictions.
        dataset (tf.data.Dataset): Dataset containing images and labels.
        count (int, optional): Number of batches to process. Defaults to -1 (process all batches).

    Returns:
        dict: Dictionary containing 'y_true' (true labels) and 'y_pred' (predicted labels).
    """
    predictions = tf.constant([], dtype=tf.int64)
    y_true = tf.constant([], dtype=tf.int64)

    for image, label in dataset.take(count):
        y_pred = tf.cast((model.predict(image, verbose=0) >= 0.5).reshape(-1), dtype=tf.int64)
        predictions = tf.concat([predictions, y_pred], -1)
        y_true = tf.concat([y_true, label], -1)
    return {'y_true': y_true, 'y_pred': predictions}


class Evaluation:
    """
    A class to evaluate a TensorFlow model on a given dataset.

    This class provides methods to generate a classification report,
    plot a confusion matrix, and access the predicted and true labels.

    Attributes:
        model (tf.keras.Model): The model to be evaluated.
        dataset (tf.data.Dataset): The dataset on which to evaluate the model.
        y_true (tf.Tensor): True labels from the dataset.
        y_pred (tf.Tensor): Predicted labels by the model.
    """
    
    def __init__(self, model, dataset, count=-1):
        """
        Initializes the Evaluation class with a model and dataset.

        Args:
            model (tf.keras.Model): Trained model to evaluate.
            dataset (tf.data.Dataset): Dataset containing images and labels.
        """
        self.model = model
        self.dataset = dataset
        self.y_true, self.y_pred = _return_pred_and_label(self.model, self.dataset, count).values()
    
    def generate_classification_report(self, digits=2):
        """
        Prints the classification report with precision, recall, and F1-score.

        Args:
            digits (int, optional): Number of decimal places to display. Defaults to 2.
        """
        print(classification_report(self.y_true, self.y_pred, digits=digits))
        
    def return_pred_and_label(self):
        """
        Returns the true and predicted labels.

        Returns:
            tuple: A tuple containing two tensors: y_true and y_pred.
        """
        return self.y_true, self.y_pred

    def plot_confusion_matrix(self, figsize=(10, 7)):
        """
        Plots the confusion matrix as a heatmap.

        Args:
            figsize (tuple, optional): Figure size for the plot. Defaults to (10, 7).
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
