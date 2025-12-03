import os
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.stats import skew, kurtosis
from src.logger import logger


def load_nifti(path: str):
    """
    Load a NIfTI image from disk and return both the SimpleITK image and its NumPy array.

    Args:
        path (str): Path to the NIfTI file.

    Returns:
        Tuple[sitk.Image, np.ndarray]: SimpleITK image and corresponding NumPy array.
    """
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    return img, arr


def intensity_features(img_arr: np.ndarray, mask_arr: np.ndarray) -> Dict[str, float]:
    """
    Compute intensity-based statistics within the masked region.

    Args:
        img_arr (np.ndarray): Image array.
        mask_arr (np.ndarray): Binary mask array.

    Returns:
        Dict[str, float]: Dictionary of intensity features.
    """
    masked = img_arr[mask_arr > 0]
    if masked.size == 0:
        return {k: np.nan for k in [
            'mean', 'std', 'min', 'max', 'median', 'skew', 'kurtosis',
            'p10', 'p25', 'p75', 'p90']}
    feats = {}
    feats['mean'] = float(np.mean(masked))
    feats['std'] = float(np.std(masked))
    feats['min'] = float(np.min(masked))
    feats['max'] = float(np.max(masked))
    feats['median'] = float(np.median(masked))
    feats['skew'] = float(skew(masked))
    feats['kurtosis'] = float(kurtosis(masked))
    feats['p10'] = float(np.percentile(masked, 10))
    feats['p25'] = float(np.percentile(masked, 25))
    feats['p75'] = float(np.percentile(masked, 75))
    feats['p90'] = float(np.percentile(masked, 90))
    return feats



def shape_features(mask_img: sitk.Image) -> Dict[str, float]:
    """
    Compute shape-based features from a binary mask.

    Args:
        mask_img (sitk.Image): SimpleITK image of the mask.

    Returns:
        Dict[str, float]: Dictionary of shape features.
    """
    if mask_img.GetPixelID() != sitk.sitkUInt8:
        mask_img = sitk.Cast(mask_img, sitk.sitkUInt8)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_img)
    feats = {}
    label_ids = stats.GetLabels()
    if len(label_ids) == 0:
        return {'volume': np.nan}
    label = label_ids[0]
    feats['volume'] = float(stats.GetPhysicalSize(label))
    return feats



def texture_features(img_arr: np.ndarray, mask_arr: np.ndarray) -> Dict[str, float]:
    """
    Compute simple texture features using a 1D co-occurrence matrix approximation.

    Args:
        img_arr (np.ndarray): Image array.
        mask_arr (np.ndarray): Binary mask array.

    Returns:
        Dict[str, float]: Dictionary of texture features.
    """
    masked = img_arr[mask_arr > 0]
    if masked.size == 0:
        return {k: np.nan for k in ['contrast', 'energy', 'homogeneity', 'correlation']}

    # Quantize intensity to 32 levels
    levels = 32
    q = np.floor((masked - masked.min()) / (masked.ptp() + 1e-8) * (levels - 1)).astype(np.int32)

    # Build 1D co-occurrence matrix
    cooc = np.zeros((levels, levels), dtype=np.float32)
    for i in range(len(q) - 1):
        cooc[q[i], q[i + 1]] += 1
    cooc /= cooc.sum() + 1e-8

    contrast = np.sum((np.arange(levels).reshape(-1, 1) - np.arange(levels).reshape(1, -1)) ** 2 * cooc)
    energy = np.sum(cooc**2)
    homogeneity = np.sum(cooc / (1.0 + np.abs(np.arange(levels).reshape(-1, 1) - np.arange(levels).reshape(1, -1))))

    # Correlation approximation
    mean_i = np.sum(np.arange(levels) * cooc.sum(axis=1))
    mean_j = np.sum(np.arange(levels) * cooc.sum(axis=0))
    std_i = np.sqrt(np.sum((np.arange(levels) - mean_i)**2 * cooc.sum(axis=1)))
    std_j = np.sqrt(np.sum((np.arange(levels) - mean_j)**2 * cooc.sum(axis=0)))
    correlation = np.sum(((np.arange(levels).reshape(-1, 1) - mean_i) *
                          (np.arange(levels).reshape(1, -1) - mean_j) * cooc))
    correlation /= (std_i * std_j + 1e-8)

    return {'contrast': float(contrast), 'energy': float(energy),
            'homogeneity': float(homogeneity), 'correlation': float(correlation)}



def extract_features_modality(img_path: str, mask_path: str) -> Dict[str, Any]:
    """
    Extract intensity, texture, and shape features from a single modality image.

    Args:
        img_path (str): Path to the image file.
        mask_path (str): Path to the mask file.

    Returns:
        Dict[str, Any]: Dictionary of extracted features.
    """
    img_arr, img = load_nifti(img_path)
    mask_arr, mask_img = load_nifti(mask_path)

    # Ensure arrays are NumPy arrays
    if isinstance(img_arr, sitk.Image):
        img_arr = sitk.GetArrayFromImage(img_arr)
    if isinstance(mask_arr, sitk.Image):
        mask_arr = sitk.GetArrayFromImage(mask_arr)

    feats = {}
    feats.update(intensity_features(img_arr, mask_arr))
    feats.update(texture_features(img_arr, mask_arr))

    # Shape features require SimpleITK image
    mask_img_sitk = mask_img if isinstance(mask_img, sitk.Image) else sitk.GetImageFromArray(mask_img)
    feats.update(shape_features(mask_img_sitk))
    return feats


def extract_case(modalities: Dict[str, str], mask_path: str) -> Dict[str, Any]:
    """
    Extract features for all modalities of a single case.

    Args:
        modalities (Dict[str, str]): Dictionary of modality names and file paths.
        mask_path (str): Path to mask file.

    Returns:
        Dict[str, Any]: Flattened dictionary of all features with modality prefix.
    """
    case_feats = {}
    for mod, path in modalities.items():
        feats = extract_features_modality(path, mask_path)
        case_feats.update({f"{mod}_{k}": v for k, v in feats.items()})
    return case_feats


def extract_dataset(dataset_dir: str, output_csv: str, modalities: List[str]) -> pd.DataFrame:
    """
    Extract radiomic features for all cases in a dataset and save as CSV.

    Args:
        dataset_dir (str): Directory containing preprocessed cases.
        output_csv (str): Path to save extracted features CSV.
        modalities (List[str]): List of modalities to process.

    Returns:
        pd.DataFrame: DataFrame of extracted features.
    """
    dataset = {}
    masks = {}

    # Build dataset dictionary with paths
    for case_id in os.listdir(dataset_dir):
        case_path = os.path.join(dataset_dir, case_id)
        if not os.path.isdir(case_path):
            continue

        dataset[case_id] = {}
        for mod in modalities:
            img_path = os.path.join(case_path, f"{mod}_preprocessed.nii.gz")
            if os.path.exists(img_path):
                dataset[case_id][mod] = img_path

        mask_path = os.path.join(case_path, "mask_resampled.nii.gz")
        if os.path.exists(mask_path):
            masks[case_id] = mask_path
        else:
            logger.info(f"Warning: mask not found for {case_id}")

    # Extract features for all cases
    rows = []
    for case_id, mod_paths in dataset.items():
        logger.info(f"Extracting features for case: {case_id}")
        if case_id not in masks:
            logger.info(f"  Skipped: no mask available.")
            continue
        feats = extract_case(mod_paths, masks[case_id])
        feats['case_id'] = case_id
        rows.append(feats)

    df = pd.DataFrame(rows)
    df = df.set_index('case_id')
    df.to_csv(output_csv)
    return df


