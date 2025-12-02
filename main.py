from src.preprocessing import preprocess_dataset
from src.extract import extract_dataset
from src.selection import run_selection_pipeline
from src.logger import logger

def run_radiomics_pipeline(
    input_dir: str,
    output_dir: str,
    modalities: list = None,
    modality_separator: str = "_",
    image_suffix: str = ".nii.gz"
):
    """
    Run the full radiomics pipeline: preprocessing, feature extraction, and feature selection.

    Args:
        input_dir (str): Path to raw dataset directory.
        output_dir (str): Path to save preprocessed images and feature CSVs.
        modalities (list): List of imaging modalities to process.
        modality_separator (str): Separator used in file names between case ID and modality.
        image_suffix (str): Suffix for image files (e.g., '.nii' or '.nii.gz').

    Returns:
        dict: Results from feature selection including final selected features DataFrame.
    """
    # 1. Preprocessing
    preprocessed = preprocess_dataset(
        input_dir=input_dir,
        output_dir=f"{output_dir}/preprocessed",
        modalities=modalities,
        spacing=(1.0, 1.0, 1.0),
        modality_separator=modality_separator,
        image_suffix=image_suffix
    )
    
    # Print preprocessed file paths for verification
    for case, paths in preprocessed.items():
        print(case)
        for key, path in paths.items():
            print("   ", key, "->", path)
    
    # 2. Feature extraction
    extract_dataset(
        dataset_dir=f"{output_dir}/preprocessed",
        output_csv=f"{output_dir}/radiomics.csv",
        modalities=modalities
    )
    
    # 3. Feature selection
    result = run_selection_pipeline(f"{output_dir}/radiomics.csv", do_bootstrap=True)
    df_final = result["df_final"]
    df_final.to_csv(f"{output_dir}/selected_features.csv", index=False)
    
    return result


def main():
    """
    Main entry point for the radiomics pipeline.
    Selects dataset and configuration parameters, then runs the pipeline.
    """
    dataset_flag = "Brats2023"  # Choose dataset identifier
    
    # Configure paths and modalities for different datasets
    if dataset_flag == "Brats2020":
        input_dir = "/hpc/ajhz839/data/BraTS2020/MICCAI_BraTS2020_TrainingData"
        output_dir = "outputs/brats2020"
        modalities = ["t1","t1ce","t2","flair"]
        modality_separator = "_"
        image_suffix = ".nii"
    elif dataset_flag == "Brats2023":
        input_dir = "/hpc/ajhz839/data/BraTS2023/train/"
        output_dir = "outputs/brats2023"
        modalities = ["t1n","t1c","t2w","t2f"]
        modality_separator = "-"
        image_suffix = ".nii.gz"
    else:
        raise ValueError("Unknown dataset flag.")
    
    # Run the full radiomics pipeline
    run_radiomics_pipeline(
        input_dir,
        output_dir,
        modalities=modalities,
        modality_separator=modality_separator,
        image_suffix=image_suffix
    )
    
    logger.info("Radiomics pipeline completed.")


if __name__ == "__main__":
    main()

