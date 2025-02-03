import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from config import cfg
from models import build_CNN


def load_preprocessed_data():
    """
    Loads the preprocessed data from numpy files and prepares the datasets.
    """
    # Load preprocessed data from numpy files
    x_train = np.load(os.path.join(cfg["data"]["processed_root"], "x_train.npy"))
    y_train = np.load(os.path.join(cfg["data"]["processed_root"], "y_train.npy"))
    x_val = np.load(os.path.join(cfg["data"]["processed_root"], "x_val.npy"))
    y_val = np.load(os.path.join(cfg["data"]["processed_root"], "y_val.npy"))
    x_test = np.load(os.path.join(cfg["data"]["processed_root"], "x_test.npy"))
    y_test = np.load(os.path.join(cfg["data"]["processed_root"], "y_test.npy"))

    return x_train, y_train, x_val, y_val, x_test, y_test

# Weighted Loss
def class_weights(y_train):
    """
    Calculates class weights to handle class imbalance.

    Parameters
    ----------
    y_train : numpy array
        Training labels

    Returns
    -------
    class_weights_dict : dict
        Mapping of class indices to weights
    """
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y = y_train
    )
    
    # Create a dictionary mapping class indices to weights
    class_weights_dict = dict(enumerate(class_weights)) # enumerate returns an iterator with index and value pairs like [(0, 0.24), (1, 0.5), (2, 0.75), (3, 1.0), (4, 1.25)] and then dictionary {0: 0.24, 1: 0.5, 2: 0.75, 3: 1.0, 4: 1.25}
    return class_weights_dict


def build_and_compile_model(input_shape, output_shape, hyperparameters):
    """
    Builds and compiles the CNN model based on the provided configuration.
    """
    model = build_CNN(
        input_shape=input_shape,
        output_shape=output_shape,
        layer_1_size=hyperparameters['layer_1_size'],
        layer_2_size=hyperparameters['layer_2_size'],
        layer_3_size=hyperparameters['layer_3_size'],
        layer_FC_size=hyperparameters['layer_FC_size'],
        dropout_rate=hyperparameters['dropout_rate']
    )

    learning_rate = float(hyperparameters['learning_rate'])
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model(model, x_train, y_train, x_val, y_val, hyperparameters):
    """
    Trains the CNN model.
    """
    
    checkpoint_dir = os.path.dirname(cfg["model"]["checkpoint_path"])
    os.makedirs(checkpoint_dir, exist_ok=True)  # Creates the directory if it doesn't exist
    
    # Get class weights
    class_weights_dict = class_weights(y_train) if cfg["model"]["class_weights"] else None
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        cfg["model"]["checkpoint_path"],
        monitor='val_accuracy',
        save_best_only=True
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=cfg["model"]["early_stopping_patience"],
        restore_best_weights=True
    )

    # Start training
    history = model.fit(
        x_train,
        y_train,
        batch_size=hyperparameters['batch_size'],
        epochs=hyperparameters['epochs'],
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, checkpoint],
        class_weight=class_weights_dict
    )

    return history


def plot_history(history, results_dir=None, model_name='model', metric=None):
    """
    Plots training and validation loss and an optional metric.

    Parameters
    ----------
    history : keras.callbacks.History
        History object returned by the `fit` method of a Keras model
    metric : str, optional
        Name of the metric to plot
    """
    
    # If results_dir is not provided, use the current working directory as the default
    if results_dir is None:
        results_dir = os.getcwd()
    
    # Create the directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    fig, ax1 = plt.subplots(figsize=(10, 8)) #figsize=(10,8)

    epoch_count=len(history.history['loss'])

    line1,=ax1.plot(range(1,epoch_count+1),history.history['loss'],label='train_loss',color='orange')
    ax1.plot(range(1,epoch_count+1),history.history['val_loss'],label='val_loss',color = line1.get_color(), linestyle = '--')
    ax1.set_xlim([1,epoch_count])
    ax1.set_ylim([0, max(max(history.history['loss']),max(history.history['val_loss']))])
    ax1.set_ylabel('loss',color = line1.get_color())
    ax1.tick_params(axis='y', labelcolor=line1.get_color())
    ax1.set_xlabel('Epochs')
    _=ax1.legend(loc='lower left')

    if (metric!=None): 
        ax2 = ax1.twinx()
        line2,=ax2.plot(range(1,epoch_count+1),history.history[metric],label='train_'+metric)
        ax2.plot(range(1,epoch_count+1),history.history['val_'+metric],label='val_'+metric,color = line2.get_color(), linestyle = '--')
        ax2.set_ylim([0, max(max(history.history[metric]),max(history.history['val_'+metric]))])
        ax2.set_ylabel(metric,color=line2.get_color())
        ax2.tick_params(axis='y', labelcolor=line2.get_color())
        _=ax2.legend(loc='upper right')
    plt.savefig(os.path.join(results_dir, f'{model_name}_training_plot.png'))
    plt.show() 


def show_confusion_matrix(conf_matrix,class_names,figsize=(10,10)):
  fig, ax = plt.subplots(figsize=figsize)
  img = ax.matshow(conf_matrix)
  tick_marks = np.arange(len(class_names))
  _=plt.xticks(tick_marks, class_names,rotation=45)
  _=plt.yticks(tick_marks, class_names)
  _=plt.ylabel('Real')
  _=plt.xlabel('Predicted')

  for i in range(len(class_names)):
    for j in range(len(class_names)):
        text = ax.text(j, i, '{0:.1%}'.format(conf_matrix[i, j]),
                       ha='center', va='center', color='w')
  plt.tight_layout()
  plt.show()
  
  
def show_classification_report_heatmap(report_df):
    """
    Plots a heatmap of the classification report.

    This function takes a DataFrame containing the classification report metrics (excluding the 'support' row) 
    and generates a heatmap to visualize the precision, recall, and F1-score for each class. The heatmap 
    helps to quickly assess the performance of the model across different classes.

    Parameters
    ----------
    report_df : pandas.DataFrame
        A DataFrame containing the classification report metrics. The DataFrame should include metrics such 
        as precision, recall, and F1-score for each class. It should be in a format where the index represents 
        class names and columns represent different metrics. The 'support' row should be excluded or not included.

    Returns
    -------
    None
        This function does not return any value. It displays the heatmap plot of the classification report.
    """
    plt.figure(figsize=(8,8))
    
    ax = sns.heatmap(report_df, annot=True, cmap='Blues')
    ax.set_yticklabels(ax.get_yticklabels(),fontsize=12, rotation=0)
    plt.title("Classification Report")
    plt.tight_layout()
    plt.show()