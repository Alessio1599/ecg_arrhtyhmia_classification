import numpy as np
from utils import plot_history
from utils import load_preprocessed_data
from utils import build_and_compile_model
from utils import train_model
from utils import load_preprocessed_data
from config import cfg  


def main():
    """
    Main function to load data, build, train the model, and plot training history.
    """
    # Prepare the data
    x_train, y_train, x_val, y_val, _, _ = load_preprocessed_data()

    # Define the input and output shapes
    input_shape = (x_train.shape[1], 1)  # Each sample has `x_train.shape[1]` features and 1 channel
    output_shape = len(np.unique(y_train))  # Number of unique classes

    # Build and compile the CNN model
    model = build_and_compile_model(input_shape, output_shape, cfg["hyperparameters"])

    # Train the model
    history = train_model(model, x_train, y_train, x_val, y_val, cfg["hyperparameters"])

    # Plot the training history
    plot_history(history, metric='accuracy')


if __name__ == "__main__":
    main()
