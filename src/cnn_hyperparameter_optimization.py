import wandb
import numpy as np
from config import cnn_sweep_cfg
from utils import load_preprocessed_data, build_and_compile_model, train_model


def sweep_train_function(config=None):
    """
    This function is called for each hyperparameter configuration during the sweep.
    It trains the model with the given hyperparameters.
    """
    # Unpack configuration from wandb
    layer_1_size = config.layer_1_size
    layer_2_size = config.layer_2_size
    layer_3_size = config.layer_3_size
    layer_FC_size = config.layer_FC_size
    dropout_rate = config.dropout_rate
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    epochs = config.epochs

    # Load preprocessed data
    x_train, y_train, x_val, y_val, _, _ = load_preprocessed_data()

    # Define input shape and output shape
    input_shape = (x_train.shape[1], 1)  # Adjust to your data
    output_shape = len(np.unique(y_train))  # Adjust based on your number of classes

    # Build and compile model with the given configuration
    model = build_and_compile_model(input_shape, output_shape, {
        'layer_1_size': layer_1_size,
        'layer_2_size': layer_2_size,
        'layer_3_size': layer_3_size,
        'layer_FC_size': layer_FC_size,
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs
    })

    # Train model
    history = train_model(model, x_train, y_train, x_val, y_val, {
        'batch_size': batch_size,
        'epochs': epochs
    })

def main():
    
    # x_train, y_train, x_val, y_val, _, _ = load_preprocessed_data()
    
    # Initialize WandB and pass the loaded config
    wandb.login()
    wandb.init(
        project="single-ECG-classification",  # Your WandB project name
        config=cnn_sweep_cfg  # Load the config from the YAML file
    )
    sweep_config = wandb.config  # Access the configuration from WandB

    # Print the config to verify
    print("Loaded WandB config:", sweep_config)

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="single-ECG-classification")

    # Start the sweep
    wandb.agent(sweep_id, function=sweep_train_function, count=5)  # Run 5 different configurations

    
    wandb.finish()

if __name__ == "__main__":
    main()
