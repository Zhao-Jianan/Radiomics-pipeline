import os
import numpy as np
import SimpleITK as sitk
from typing import Dict, Tuple
from src.logger import logger


def load_image(path: str) -> sitk.Image:
    """
    Load a NIfTI image from disk.

    Args:
        path (str): Path to the image file.

    Returns:
        sitk.Image: Loaded SimpleITK image.
    """
    return sitk.ReadImage(path)


def save_image(img: sitk.Image, path: str):
    """
    Save a SimpleITK image to disk, creating directories if necessary.

    Args:
        img (sitk.Image): Image to save.
        path (str): Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sitk.WriteImage(img, path)



def resample_image(
    image: sitk.Image,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    interpolator=sitk.sitkBSpline,
) -> sitk.Image:
    """
    Resample an image to new voxel spacing using the specified interpolator.

    Args:
        image (sitk.Image): Input image.
        spacing (Tuple[float,float,float]): Desired output spacing.
        interpolator: SimpleITK interpolator type.

    Returns:
        sitk.Image: Resampled image.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(original_size[i] * (original_spacing[i] / spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetSize(new_size)
    resampler.SetInterpolator(interpolator)

    return resampler.Execute(image)


def resample_mask(mask: sitk.Image, spacing=(1.0, 1.0, 1.0)) -> sitk.Image:
    """
    Resample a mask using nearest-neighbor interpolation to preserve labels.

    Args:
        mask (sitk.Image): Input mask.
        spacing (Tuple[float,float,float]): Desired output spacing.

    Returns:
        sitk.Image: Resampled mask.
    """
    return resample_image(mask, spacing=spacing, interpolator=sitk.sitkNearestNeighbor)



def zscore_normalize(image: sitk.Image) -> sitk.Image:
    """
    Apply Z-score normalization to an image.

    Args:
        image (sitk.Image): Input image.

    Returns:
        sitk.Image: Normalized image.
    """
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    mean = arr.mean()
    std = arr.std() + 1e-6  # Avoid division by zero
    arr = (arr - mean) / std

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    return out



def preprocess_case(
    images: Dict[str, str],
    mask_path: str,
    output_dir: str,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, str]:
    """
    Preprocess all modalities and mask for a single patient.

    Args:
        images (Dict[str,str]): Dictionary of modality name -> image path.
        mask_path (str): Path to segmentation mask.
        output_dir (str): Directory to save preprocessed outputs.
        spacing (Tuple[float,float,float]): Desired voxel spacing.

    Returns:
        Dict[str,str]: Dictionary of saved file paths (modalities + mask).
    """
    os.makedirs(output_dir, exist_ok=True)
    processed_paths = {}

    # Resample and save mask
    mask = load_image(mask_path)
    mask_resampled = resample_mask(mask, spacing=spacing)
    mask_save = os.path.join(output_dir, "mask_resampled.nii.gz")
    save_image(mask_resampled, mask_save)
    processed_paths["mask"] = mask_save

    # Resample, normalize, and save each modality
    for modality, path in images.items():
        img = load_image(path)
        img = resample_image(img, spacing=spacing)
        img = zscore_normalize(img)

        out_path = os.path.join(output_dir, f"{modality}_preprocessed.nii.gz")
        save_image(img, out_path)
        processed_paths[modality] = out_path

    return processed_paths



def preprocess_dataset(
    input_dir: str,
    output_dir: str,
    modalities: list,
    spacing: Tuple[float,float,float] = (1.0,1.0,1.0),
    modality_separator="_",
    image_suffix=".nii"
) -> Dict[str, Dict[str, str]]:
    """
    Scan dataset directory, preprocess all cases, and return paths to preprocessed images.

    Handles both BraTS 2020 and 2023 naming conventions.

    Args:
        input_dir (str): Root directory of raw dataset.
        output_dir (str): Directory to save preprocessed data.
        modalities (list): List of modalities to process.
        spacing (Tuple[float,float,float]): Desired voxel spacing.
        modality_separator (str): Separator used in file names.
        image_suffix (str): File extension (e.g., .nii or .nii.gz).

    Returns:
        Dict[str, Dict[str,str]]: Nested dictionary: {case_id: {modality/mask: path}}
    """
    dataset = {}
    masks = {}

    # Build dataset dictionary
    for case_id in os.listdir(input_dir):
        case_path = os.path.join(input_dir, case_id)
        if not os.path.isdir(case_path):
            continue

        dataset[case_id] = {}
        # Locate modality images
        for mod in modalities:
            img_name = f"{case_id}{modality_separator}{mod}{image_suffix}"
            img_path = os.path.join(case_path, img_name)
            if os.path.exists(img_path):
                dataset[case_id][mod] = img_path

        # Locate mask
        mask_name = f"{case_id}{modality_separator}seg{image_suffix}"
        mask_path = os.path.join(case_path, mask_name)
        if os.path.exists(mask_path):
            masks[case_id] = mask_path
        else:
            logger.info(f"Warning: Mask not found for {case_id}, skipping this case.")
            continue  # Skip cases without mask

    # Preprocess all cases
    results = {}
    for case_id in dataset:
        if case_id not in masks:
            continue

        case_output = os.path.join(output_dir, case_id)
        os.makedirs(case_output, exist_ok=True)

        results[case_id] = preprocess_case(
            images=dataset[case_id],
            mask_path=masks[case_id],
            output_dir=case_output,
            spacing=spacing
        )
        logger.info(f"Preprocessed case: {case_id}")

    logger.info("Preprocessing completed.")
    return results
