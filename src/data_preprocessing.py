import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path
import logging
from config import cfg

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

def load_data(data_dir=None):
    """
    Reads ECG data from CSV files and returns training, validation, and test sets.
    """
    if data_dir is None:
        data_dir = Path(cfg["data"]["raw_root"]).resolve()

    data_dir = Path(data_dir)
    
    # Read CSV files
    try:
        train_df = pd.read_csv(data_dir / "mitbih_train.csv", header=None)
        test_df = pd.read_csv(data_dir / "mitbih_test.csv", header=None)
    except FileNotFoundError as e:
        logging.error(f"Data file not found: {e}")
        raise

    # Extract features and labels
    x_train, y_train = train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values
    x_test, y_test = test_df.iloc[:, :-1].values, test_df.iloc[:, -1].values

    # Split training set into validation and test
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=cfg["data"]["val_size"], random_state=42, stratify=y_train
    )
    
    logging.info(f"Data loaded. Train size: {len(x_train)}, Validation size: {len(x_val)}, Test size: {len(x_test)}")
    return x_train, y_train, x_val, y_val, x_test, y_test

def preprocess_data(x_train, x_val, x_test, method="standard"):
    """
    Normalizes ECG data using StandardScaler or MinMaxScaler.
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler(feature_range=(0, 1))
    else:
        logging.error(f"Unsupported scaling method: {method}")
        raise ValueError("Unsupported scaling method. Choose 'standard' or 'minmax'.")

    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    logging.info(f"Data scaled using {method} method.")
    return x_train_scaled, x_val_scaled, x_test_scaled

def save_preprocessed_data(x_train, y_train, x_val, y_val, x_test, y_test, processed_dir=None):
    """
    Saves preprocessed data as numpy arrays in the processed directory.
    """
    if processed_dir is None:
        processed_dir = Path(cfg["data"]["processed_root"]).resolve()

    processed_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    np.save(processed_dir / "x_train.npy", x_train)
    np.save(processed_dir / "y_train.npy", y_train)
    np.save(processed_dir / "x_val.npy", x_val)
    np.save(processed_dir / "y_val.npy", y_val)
    np.save(processed_dir / "x_test.npy", x_test)
    np.save(processed_dir / "y_test.npy", y_test)

    logging.info(f"Preprocessed data saved to {processed_dir}")

if __name__ == "__main__":
    # Load, preprocess, and save data
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(parent_dir, "data/raw")
    
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(data_dir)
    x_train, x_val, x_test = preprocess_data(x_train, x_val, x_test, method=cfg["data"].get("scaling", "standard"))
    save_preprocessed_data(x_train, y_train, x_val, y_val, x_test, y_test)
