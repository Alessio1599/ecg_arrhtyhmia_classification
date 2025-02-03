import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import tensorflow as tf
from config import cfg
from utils import show_confusion_matrix
from utils import show_classification_report_heatmap

def get_prob_predictions(model, x_test):
    """
    Returns the predicted probabilities for each class.
    """
    y_test_conf_pred_probs = model.predict(x_test) # Predicted probabilities # Get probability scores for all classes
    y_test_pred = np.argsort(y_test_conf_pred_probs, axis=1)[:, -1]  # Predicted classes, the one with the highest probability
    return y_test_conf_pred_probs, y_test_pred

def main():
    # Load the model
    model = tf.keras.models.load_model(cfg["model"]["checkpoint_path"])

    # Load preprocessed data
    x_test = np.load(os.path.join(cfg["data"]["processed_root"], "x_test.npy"))
    y_test = np.load(os.path.join(cfg["data"]["processed_root"], "y_test.npy"))

    # Get predicted probabilities and classes
    y_test_conf_pred_probs, y_test_pred = get_prob_predictions(model, x_test)

    # Get the confusion matrix
    cm = confusion_matrix(y_test, y_test_pred, normalize='true')
    show_confusion_matrix(cm, cfg["data"]["class_names"])

    # Get the classification report
    report = classification_report(
        y_test, 
        y_test_pred,
        target_names=cfg["data"]["class_names"],
        output_dict=True
    )
    # Convert classification report to DataFrame
    report_df = pd.DataFrame(report).iloc[:-1, :].T # .iloc[:-1, :] to exclude support
    show_classification_report_heatmap(report_df)

if __name__ == "__main__":
    main()