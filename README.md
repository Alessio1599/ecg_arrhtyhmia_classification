# Classification of arrhythmias from single ECG signals  using a Convolutional Neural Network

## Project Overview
This repository contains the implementation of deep learning models for the classification of cardiac arrhythmias from single-lead ECG signals. The models are trained on the MIT-BIH Arrhythmia Dataset, utilizing Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) for classification. The project includes scripts for data preprocessing, training, evaluation, and hyperparameter tuning using Weights & Biases (WandB).


## Dataset
- [MIT-BIH Arrhythmia Dataset, Kaggle dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)

The dataset consists of 48 half-hour ECG recordings sampled at 360 Hz, collected from 47 different patients.
The dataset contains 5 classes of beats:
1. Normal Beats
2. Supraventricular Ectopic Beats
3. Ventricular Ectopic Beats
4. Fusion Beats
5. Unclassifiable Beats

## Project Structure
```
ecg_arrhythmia_classification/
├── README.md                               # Project documentation
├── configs                                 # Configuration files
│   ├── cnn_sweep_config.yml               # Hyperparameter tuning config
│   └── config.yml                          # General model & training config
├── models                                  # Saved models
│   └── cnn                                # CNN model checkpoints
│       └── best_cnn_model_v1_script.keras # Best trained CNN model
├── notebooks                               # Jupyter notebooks for analysis
│   ├── eda.ipynb                          # Exploratory Data Analysis (EDA)
│   └── single_ecg_classification_COLAB_KAGGLE.ipynb  # Training on Colab/Kaggle
├── report.pdf                              # Detailed report with CNN & RNN results
└── src                                     # Source code
    ├── cnn_hyperparameter_optimization.py  # CNN hyperparameter tuning
    ├── cnn_train.py                        # CNN training script
    ├── config.py                           # Configuration loader
    ├── data_preprocessing.py               # Data preprocessing
    ├── eval.py                             # Model evaluation
    ├── models.py                           # CNN model architecture
    └── utils.py                            # Utility functions

```

## Installation & Setup
```bash
git clone https://github.com/Alessio1599/ecg_arrhtyhmia_classification.git
cd ecg_arrhtyhmia_classification
```

### Install dependendices
```bash
pip install -r requirements.txt
```

### Configuration
- Before running any scripts, you need to set up the **config.yml** and **csnn_seep_config.yml**
- Adjust the configurations to specify the paths to the datasets as well as the hyperparameters to tune 

### Data Preprocessing
To preprocess the raw dataset and save the processed .npy files:
```bash 
python src/data_preprocessing.py
```

### Training the CNN Model

To train the Convolutional Neural Network (CNN) model:
```bash
python src/cnn_train.py
```
The model and logs will be saved in the models/cnn/ directory.

### Hyperparameter Tuning with WandB
To run a hyperparameter sweep using Weights & Biases:
```bash
wandb sweep configs/cnn_sweep_config.yml
wandb agent <SWEEP_ID>
```
Alternatively, you can run:
```bash 
python src/cnn_hyperparameter_optimization.py
```

### Evaluating the Model
To evaluate the trained model on the test set:
```bash
python src/eval.py
```

### Steps overview
- Splitting dataset into training, validation, test sets
- Normalization
- Model architecture definition
- Hyperparameter tuning using WandB
- Model training
- Performance evaluation (confusion matrix, Precision and Recall)
