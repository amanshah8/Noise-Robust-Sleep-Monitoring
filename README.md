# Snore Detection using Machine Learning and Deep Learning

This repository contains a complete audio classification workflow for detecting **snore** and **non-snore** sounds from short audio clips. The project compares traditional machine learning models, deep learning models, noise-augmented training, and model explainability techniques to evaluate how well different approaches perform under clean and noisy conditions.

The main goal is to build a reliable binary audio classifier that can identify snoring sounds even when background noise is present. The project uses waveform preprocessing, MFCC feature extraction, log-mel spectrograms, ESC-50 noise augmentation, YAMNet-based embeddings, classical ML models, neural networks, SHAP, and Grad-CAM visualisations.

---

## Project Overview

Snore detection is a practical audio classification problem where the model must distinguish between snoring and non-snoring sounds. Real-world audio is rarely clean, so this project also tests model robustness by adding environmental noise at different signal-to-noise ratio levels.

The repository includes several experiments:

- Audio waveform and spectrogram visualisation
- Noise augmentation using ESC-50 environmental sounds
- MFCC-based feature extraction
- Logistic Regression, Random Forest, and SVM classifiers
- CNN models trained directly on log-mel spectrograms
- YAMNet embedding-based deep learning models
- Extreme noise training experiments
- SHAP explanations for feature importance
- Grad-CAM heatmaps for CNN interpretability

---

## Repository Structure

```text
.
├── audio_vis.ipynb              # Audio waveform, spectrogram, and playback visualisation
├── ndl_lr_ny.ipynb              # Logistic Regression using MFCC features, no YAMNet
├── ndl_rf_ny.ipynb              # Random Forest using MFCC/audio features, no YAMNet
├── ndl_svm_ny.ipynb             # SVM with GridSearchCV using MFCC features, no YAMNet
├── ndl_rf_svm_lr_yn.ipynb       # Comparison of Random Forest, SVM, and Logistic Regression with YAMNet features
├── dl_ny.ipynb                  # End-to-end CNN using log-mel spectrograms, no YAMNet
├── dl_yn.ipynb                  # Deep learning model using YAMNet-based features
├── dl_noise_5.ipynb             # Extreme noise training experiments using YAMNet + MLP
├── dl_shap.ipynb                # SHAP and Grad-CAM explainability experiments
└── README.md
```

---

## Dataset

The project uses a binary snore classification dataset with two classes:

```text
/content/drive/MyDrive/snore_data/Snoring Dataset/1   # Snore audio files
/content/drive/MyDrive/snore_data/Snoring Dataset/0   # Non-snore audio files
```

The experiments were run with:

- **500 snore audio files**
- **500 non-snore audio files**
- **2,000 ESC-50 noise audio files** for noise augmentation

The ESC-50 dataset is used to simulate real-world noisy environments by mixing background sounds into the original audio clips at different SNR levels.

---

## Methodology

### 1. Audio Preprocessing

Each audio file is loaded as mono audio at **16 kHz**. The clips are padded or trimmed to a fixed duration, usually **3 seconds**, to keep input dimensions consistent across all models.

### 2. Noise Augmentation

To improve robustness, random background noise is mixed with the original audio using ESC-50 samples. The project tests multiple signal-to-noise ratio ranges, including moderate, heavy, severe, very-heavy, and extreme noise conditions.

### 3. Feature Extraction

Different notebooks use different feature extraction approaches:

- **MFCC mean and standard deviation features** for traditional ML models
- **Log-mel spectrograms** for CNN-based models
- **YAMNet embeddings/statistical features** for transfer-learning-based models

### 4. Model Training

The project compares both traditional and deep learning approaches:

- Logistic Regression
- Random Forest
- Support Vector Machine with GridSearchCV
- CNN trained on log-mel spectrograms
- MLP trained on YAMNet-based audio embeddings

### 5. Evaluation

Models are evaluated on both clean and noisy test sets using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

---

## Key Results

### Classical Machine Learning Models with YAMNet Features

| Model | Test Split | Accuracy | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | Clean | 0.985 | 0.985 | 0.999 |
| Random Forest | Clean | 0.965 | 0.966 | 0.997 |
| SVM | Clean | 0.930 | 0.933 | 0.994 |
| Random Forest | Noisy | 0.895 | 0.894 | 0.960 |
| Logistic Regression | Noisy | 0.870 | 0.867 | 0.954 |
| SVM | Noisy | 0.860 | 0.860 | 0.957 |

The best clean-test result was achieved by **Logistic Regression with YAMNet features**, reaching **98.5% accuracy** and **0.999 ROC-AUC**.

### MFCC-Based Traditional Models without YAMNet

| Model | Clean Accuracy | Noisy Accuracy | Clean ROC-AUC | Noisy ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.850 | 0.735 | 0.948 | 0.796 |
| Random Forest | 0.790 | 0.765 | 0.954 | 0.855 |
| SVM | 0.940 | 0.800 | 0.991 | 0.895 |

Among the no-YAMNet traditional models, **SVM with MFCC features** performed strongest on clean audio with **94% accuracy** and remained reasonably robust on noisy audio.

### Extreme Noise Training with YAMNet + MLP

| Training Regime | Clean Accuracy | Noisy Accuracy | Clean ROC-AUC | Noisy ROC-AUC |
|---|---:|---:|---:|---:|
| Baseline-Med SNR 8–14 dB | 0.975 | 0.855 | 0.999 | 0.913 |
| Extreme SNR -10 to -6 dB | 0.940 | 0.790 | 0.989 | 0.853 |
| Heavy SNR 0–7 dB | 0.930 | 0.845 | 0.983 | 0.932 |
| Severe SNR -5–0 dB | 0.965 | 0.880 | 0.993 | 0.913 |
| Very-Heavy SNR 0–5 dB | 0.920 | 0.845 | 0.994 | 0.934 |

The **Severe noise training regime** produced the best noisy-test accuracy at **88%**, showing that training with difficult noise conditions can improve real-world robustness.

---

## Explainability

The project also includes explainability experiments to make the model behaviour easier to interpret.

### SHAP

SHAP is used to explain the feature contributions of the pretrained MLP model using YAMNet statistical inputs. This helps identify which extracted audio features influence the snore/non-snore prediction most strongly.

### Grad-CAM

Grad-CAM is applied to CNN models trained on log-mel spectrograms. The heatmaps show which regions of the spectrogram are most important for the model's decision, helping connect predictions back to visible audio patterns.

---

## Technologies Used

- Python
- Google Colab
- TensorFlow / Keras
- Scikit-learn
- Librosa
- NumPy
- Pandas
- Matplotlib
- SHAP
- YAMNet
- Kaggle API
- ESC-50 environmental noise dataset

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 2. Open the notebooks in Google Colab

Most notebooks are designed for Google Colab because they use Google Drive mounting and Kaggle dataset downloads.

### 3. Upload Kaggle API credentials

Download `kaggle.json` from your Kaggle account settings and upload it when prompted by the notebook.

### 4. Prepare Google Drive folders

Expected folder structure:

```text
/content/drive/MyDrive/snore_data/Snoring Dataset/1
/content/drive/MyDrive/snore_data/Snoring Dataset/0
/content/drive/MyDrive/noise_data/audio/audio
/content/drive/MyDrive/snore_models
```

### 5. Run notebooks from top to bottom

Recommended order:

1. `audio_vis.ipynb`
2. `ndl_lr_ny.ipynb`
3. `ndl_rf_ny.ipynb`
4. `ndl_svm_ny.ipynb`
5. `ndl_rf_svm_lr_yn.ipynb`
6. `dl_ny.ipynb`
7. `dl_yn.ipynb`
8. `dl_noise_5.ipynb`
9. `dl_shap.ipynb`

---

## Main Findings

- YAMNet-based features significantly improved model performance compared with basic MFCC-only models.
- Logistic Regression with YAMNet features achieved the strongest clean-test result.
- Random Forest and Logistic Regression remained strong under noisy conditions when trained with robust embeddings.
- SVM performed well without YAMNet, especially on clean MFCC features.
- CNNs trained directly on log-mel spectrograms were less stable in this setup and required more tuning.
- Training with severe background noise improved robustness on noisy test audio.
- SHAP and Grad-CAM helped make model predictions more interpretable.

---

## Future Improvements

- Convert notebooks into reusable Python scripts
- Add a single training pipeline with configuration files
- Save trained models and scalers in a dedicated `models/` directory
- Add inference code for testing new audio files
- Build a simple web or mobile demo for uploading and classifying audio
- Experiment with larger audio datasets and more diverse background noise
- Fine-tune deep learning architectures for better spectrogram-based performance
- Add automated tests and reproducible environment files

---

## Conclusion

This project demonstrates a complete experimental pipeline for snore detection using audio machine learning. By comparing classical models, deep learning models, transfer-learning features, noise augmentation, and explainability tools, the work shows that robust snore classification is achievable even under noisy real-world conditions.

The strongest overall results came from models using **YAMNet-based audio features**, especially Logistic Regression and Random Forest, while noise-augmented training improved resilience against environmental sound interference.
