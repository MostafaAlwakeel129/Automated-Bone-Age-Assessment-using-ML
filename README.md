# Automated Bone Age Assessment using Machine Learning

Automated pediatric bone age assessment from hand X-rays using classical computer vision and gradient boosting. Built with a production-ready, config-driven training pipeline.

---

## Overview

Bone age assessment is a standard clinical procedure used to evaluate skeletal maturity in children — informing diagnoses of growth disorders, hormonal conditions, and developmental delays. This project automates that process using the [RSNA Bone Age dataset](https://www.kaggle.com/datasets/kmader/rsna-bone-age), replacing manual radiologist readings with an ML pipeline built on handcrafted image features and a Random Forest regressor.

The pipeline predicts bone age (in months) from a pediatric hand X-ray with a **Mean Absolute Error of ~X months** on held-out test data.

---

## How It Works

### 1. Feature Extraction (`extract_features.py`)

Raw X-rays go through a multi-stage processing pipeline before any learning happens:

**Preprocessing & ROI Isolation**
- CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied to normalize contrast across images
- Otsu thresholding + morphological operations isolate the hand from background noise
- Contour analysis finds and crops the dominant hand region

**Region-Specific Feature Extraction**
Two anatomical sub-regions are independently analyzed — the **knuckles** and the **wrist** — since both carry distinct developmental signals. For each:

- **SIFT Bag of Visual Words (BoVW):** A visual dictionary of 150 clusters is learned from a 500-image sample using MiniBatch K-Means. Each image region is encoded as a normalized histogram over this dictionary (150 features)
- **HOG (Histogram of Oriented Gradients):** Captures local texture and edge structure (20 features)
- **Global Intensity Statistics:** Mean, standard deviation, and 16-bin intensity histogram (18 features)

**Final Feature Vector: 189 dimensions per image** (including a binary sex flag)

> This step is designed to run in a cloud environment (Kaggle/Colab) due to dataset size. The resulting `.npy` arrays are saved locally for training.

---

### 2. Training Pipeline (`train.py`)

The training pipeline is fully config-driven and designed for reproducibility:

```
train.py
├── Loads config.yaml and model_config.yaml
├── Creates a timestamped experiment directory under ./models/
├── Copies config files into that directory (snapshot of exact settings used)
├── Loads and scales features (StandardScaler)
├── Trains a Random Forest on the training split
└── Evaluates on test split → saves model, scaler, and metrics
```

Every run produces a self-contained experiment folder:

```
models/
└── 2026_05_15_14_30/
    ├── baseline_rf.pkl       # Trained model
    ├── feature_scaler.pkl    # Fitted scaler (required for inference)
    ├── metrics.txt           # MAE and R² on test set
    ├── config.yaml           # Exact data config used
    └── model_config.yaml     # Exact model config used
```

---

## Project Structure

```
.
├── config/
│   ├── config.yaml           # Data paths and split parameters
│   └── model_config.yaml     # Model hyperparameters
│
├── processed_data/           # Output of extract_features.py
│   ├── X_sift_full.npy
│   └── y_sift_full.npy
│
├── models/                   # Auto-created; one folder per training run
│
├── extract_features.py       # Feature engineering (run in cloud)
├── train.py                  # Main training entrypoint
├── boneage_utils_data.py     # Data loading and preprocessing
├── boneage_utils_models.py   # Model definitions
├── boneage_utils_eval.py     # Evaluation and artifact saving
├── boneage_utils_helper.py   # Config loading and experiment tracking
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- The processed `.npy` feature arrays (see [Feature Extraction](#feature-extraction) below)

### Installation

```bash
git clone https://github.com/your-username/Automated-Bone-Age-Assessment-using-ML.git
cd Automated-Bone-Age-Assessment-using-ML
pip install -r requirements.txt
```

### Configuration

Edit `config/config.yaml` to point to your data:

```yaml
data_paths:
  X_data: "./processed_data/X_sift_full.npy"
  y_data: "./processed_data/y_sift_full.npy"

save_dir: "./models"

split_params:
  test_size: 0.2
  random_state: 42
```

Edit `config/model_config.yaml` to tune the model:

```yaml
random_forest:
  n_estimators: 10
  max_depth: null
  min_samples_split: 2
  random_state: 42
  n_jobs: -1
```

### Run Training

```bash
python train.py
```

Output:

```
Starting pipeline
Created experiment tracking folder: ./models/2026_05_15_14_30

Loading features from: ./processed_data/X_sift_full.npy
Loading labels from: ./processed_data/y_sift_full.npy

Splitting data into train and test sets
Scaling features
Data ready: 10400 train samples, 2600 test samples.

Training Random Forest model
Training complete.

Evaluating model:
Mean Absolute Error: XX.XX months
R-squared Score:     0.XXXX

Saving model and scaler...
Pipeline complete. Results saved in: ./models/2026_05_15_14_30
```

---

## Feature Extraction

Because the raw RSNA dataset is large (~12k images, several GB), the feature extraction step is designed to run in the cloud.

**On Kaggle or Colab:**

1. Add the [RSNA Bone Age dataset](https://www.kaggle.com/datasets/kmader/rsna-bone-age) to your environment
2. Update the paths at the top of `extract_features.py`:
   ```python
   TRAIN_CSV_PATH = '/kaggle/input/...'
   TRAIN_IMG_DIR  = '/kaggle/input/...'
   ```
3. Run the script — it will produce `X_sift_full.npy` and `y_sift_full.npy`
4. Download these two files into your local `processed_data/` folder

---

## Requirements

```
numpy
pandas
scikit-learn
joblib
opencv-python
scikit-image
PyYAML
tqdm
matplotlib
seaborn
```

---

## Dataset

**RSNA Pediatric Bone Age Challenge (2017)**
- ~12,600 hand X-rays of pediatric patients
- Labels: bone age in months + biological sex
- Source: [Kaggle — RSNA Bone Age](https://www.kaggle.com/datasets/kmader/rsna-bone-age)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
