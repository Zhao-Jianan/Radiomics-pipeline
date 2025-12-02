# Radiomics Feature Extraction Pipeline

This repository provides a Python-based pipeline for **radiomics feature extraction and selection** from multi-modal medical images (e.g., MRI). The pipeline supports preprocessing, feature extraction, and unsupervised feature selection, designed for use in research and machine learning applications.

---

## Features

1. **Preprocessing**
   - Resampling images and masks to uniform voxel spacing.
   - Intensity normalization (Z-score).
   - Handles multiple modalities (T1, T1ce, T2, FLAIR).

2. **Feature Extraction**
   - Intensity features: mean, standard deviation, min, max, median, skewness, kurtosis, percentiles.
   - Shape features: volume (mask surface area calculation optional).
   - Simple texture features (GLCM-like) using NumPy: contrast, energy, homogeneity, correlation.

3. **Feature Selection**
   - Missing value imputation using median.
   - Low-variance filtering.
   - High-correlation feature removal.
   - Optional bootstrap-based stability selection.

4. **Logging**
   - Centralized logging to console and optional file.
   - Tracks preprocessing and feature extraction progress.

---

## Installation

1. Clone the repository:

```bash
git clone <repository_url>
cd <repository_directory>
```

2. Create and activate a Python environment:
```bash
conda create -n radiomics python=3.11 -y
conda activate radiomics
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

**Dependencies include**: `numpy`, `pandas`, `SimpleITK`, `scipy`.


## Usage
The main pipeline is defined in `main.py`. The workflow includes preprocessing, feature extraction, and feature selection.

```bash
python main.py
```

## Configurable Parameters
- **dataset_flag**: "Brats2020" or "Brats2023".
- **input_dir**: Path to raw MRI dataset.
- **output_dir**: Directory to save preprocessed images and feature CSVs.
- **modalities**: List of image modalities.
- **modality_separator**: Separator used in file names.
- **image_suffix**: File extension (e.g., .nii, .nii.gz).


## Outputs
### Preprocessed images
- Saved under `output_dir/<case_id>/`.
- Format: `{modality}_preprocessed.nii.gz`.

### Radiomics feature table
- `radiomics.csv`: Contains all extracted features for each case.
- Columns: `case_id`, `t1_mean`, `t1_std`, ..., `flair_correlation`.

### Selected features
- `selected_features.csv`: Subset of stable and non-redundant features after feature selection.
