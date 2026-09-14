
# Sensor-based Human Activity Classification using Deep Learning  
*Team Project - DL_Project_Team27*

---

##  1. Problem Description

**Objective**  
This project focuses on multi-class classification of human activity using data collected from PIR (Passive Infra-Red) motion sensors. We aim to experiment with deep learning models to accurately predict one of three class labels corresponding to human presence/activity states and report accuracy for each of the models.

---

## Dataset Overview

The **PIRvision dataset** contains occupancy detection data captured using synchronized Low-Energy Electronically-Chopped Passive Infra-Red (PIR) sensor nodes. The sensors were deployed in residential and office environments.

- Each observation records **4 seconds of human activity** within the sensor’s Field-of-View (FoV).
- The dataset contains raw features (motion, timestamps, signal values) and class labels for three distinct activity states.

---

##  Project Workflow

1. **Exploratory Data Analysis (EDA)** – Understand trends, patterns, class distributions, and correlations.
2. **Feature Engineering** – Extract meaningful features from raw sensor readings.
3. **SMOTE (Synthetic Minority Over-sampling Technique)** – Handle data imbalance by synthesizing minority class samples.
4. **Dataset Preparation** – Created three datasets:
   -  cleaned dataset with rephrased labelling and timestamp removal(Unbalanced)
   -  Feature engineered dataset (Unbalanced)
   -  SMOTE-augmented datasets (Balanced)
5. **Model Training & Evaluation** – Perform **Stratified K-Fold Cross Validation** on each dataset using:
   - MLP (Multi-layer Perceptron)
   - LSTM (Long Short-Term Memory)
   - 1D-CNN (Convolutional Neural Network)
6. **Model Checkpointing** – Save best-performing models for each dataset-model combination (3 datasets × 3 models = 9 total).
7. **Metrics Reporting** – Log and save:
   - Mean Accuracy
   - Precision, Recall, F1-score
   - Confusion Matrix
   - Classification Report
   - Mean Macro F1 score

---

## 💻 2. Language and Libraries

- **Language:** Python 3
- **Libraries Used:** pandas, numpy, matplotlib, seaborn, imblearn, torch, sklearn, datetime, warnings, re, ast

---

## 3. Directory Structure

```
DL_Project_Team27
│
├── code.ipynb                        # Main notebook with all code and results
│
├── Dataset/                          # Sensor dataset directory
│   ├── PIRVISION_OFFICE_DATASET1.csv
│   └── PIRVISION_OFFICE_DATASET2.csv
│
├── model_paths/                      # Saved checkpoints
│   ├── lstm/
│   ├── cnn/
│   └── mlp/
│
└── results/                          # Experiment results
    ├── *.csv                         # 9 CSV files (3 models × 3 datasets)
```

---

##  4. Instructions for Execution

###  Training (Run Full Experiment Pipeline)

To train all models on all datasets:

1. Ensure the folder structure is as described above.
2. Launch the notebook:

```bash
jupyter notebook code.ipynb
```

3. Run all the cells sequentially (top to bottom).  
   This will:
   - Preprocess and generate all 3 datasets.
   - Train MLP, LSTM, and CNN models with stratified k-fold CV.
   - Save best model checkpoints and evaluation results.

---

###  Testing (Inference Mode)

To perform **testing only** using saved models:

- In the notebook, look for **Markdown cells labeled "For Inference / Testing Only"**.
- Run **only those cells** to load saved checkpoints and perform predictions(accuracy) without retraining.

---

##  Results Overview


>  Check the `results/` folder for individual `.csv` reports on  mean accuracy, mean macro F1-scores for each model and dataset combination which are used as baseline for saving best model checkpoint.

---

##  Checkpointing

Best model weights pertaining to each model and each dataset(created during preprocessing) are automatically saved in:

```
model_paths/
├── mlp/
     |-f"mlp_data_{dataset_type}_best_lr{lr}_bs{batch_size}_hidden_dim[{hidden_str}]_dropout{dropout}.pth"   ## 3 files each
├── lstm/
     |-f"lstm_data_{dataset_type}_best_lr{lr}_bs{batch_size}_lstm_layers{lstm_layers}_hidden_dim{hidden_dim}_dropout{dropout}.pth"  ## 3 files each
|── cnn/
     |-f"cnn_data_{dataset_type}_best_lr{lr}_bs{batch_size}_convfilter_{conv_filters}_kernel_size_{kernel_size}_fcsize_{fc_size}_dropout{dropout}.pth" #3 f
```

These can be used for **quick running to print accuracies** .

---

