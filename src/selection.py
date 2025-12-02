import numpy as np
import pandas as pd
from typing import List, Dict

def load_feature_table(csv_path: str) -> pd.DataFrame:
    """
    Load a radiomics CSV table.
    Assumes first column is 'case_id', remaining columns are features.

    Args:
        csv_path (str): Path to CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    df = pd.read_csv(csv_path)
    return df



def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing feature values with the median of each feature.

    Args:
        df (pd.DataFrame): Input dataframe with features.

    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    feature_cols = df.columns[1:]
    df_filled = df.copy()
    df_filled[feature_cols] = df_filled[feature_cols].fillna(df_filled[feature_cols].median())
    return df_filled



def remove_low_variance(df: pd.DataFrame, threshold: float = 1e-5):
    """
    Remove features with variance below a threshold.

    Args:
        df (pd.DataFrame): Input dataframe.
        threshold (float): Minimum variance to keep a feature.

    Returns:
        df_new (pd.DataFrame): DataFrame after removing low-variance features.
        kept (List[str]): List of kept feature names.
    """
    feature_cols = df.columns[1:]
    variances = df[feature_cols].var()
    kept = variances[variances > threshold].index.tolist()
    df_new = pd.concat([df.iloc[:, :1], df[kept]], axis=1)
    return df_new, kept



def remove_high_correlation(df: pd.DataFrame, threshold: float = 0.9):
    """
    Remove highly correlated features using Spearman correlation.

    Args:
        df (pd.DataFrame): Input dataframe.
        threshold (float): Correlation threshold for dropping features.

    Returns:
        df_new (pd.DataFrame): DataFrame after removing highly correlated features.
        kept (List[str]): List of kept feature names.
    """
    feature_cols = df.columns[1:]
    corr = df[feature_cols].corr(method='spearman').abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    kept = [f for f in feature_cols if f not in to_drop]

    df_new = pd.concat([df.iloc[:, :1], df[kept]], axis=1)
    return df_new, kept



def bootstrap_stable_features(df: pd.DataFrame, n_bootstrap: int = 50, threshold: float = 0.7, random_state: int = 42) -> List[str]:
    """
    Compute feature stability via bootstrap. Keep features that survive correlation filtering
    in at least 'threshold' fraction of bootstrap iterations.

    Args:
        df (pd.DataFrame): Input dataframe after missing value and variance filtering.
        n_bootstrap (int): Number of bootstrap iterations.
        threshold (float): Minimum stability frequency to retain a feature.
        random_state (int): Random seed.

    Returns:
        List[str]: Stable feature names.
    """
    np.random.seed(random_state)
    feature_cols = df.columns[1:]
    n_samples = df.shape[0]
    freq = pd.Series(0, index=feature_cols)

    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        df_bs = df.iloc[idx, 1:]
        corr = df_bs.corr(method='spearman').abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
        kept = [f for f in feature_cols if f not in to_drop]
        freq[kept] += 1

    freq /= n_bootstrap
    stable_features = freq[freq >= threshold].index.tolist()
    return stable_features



def run_selection_pipeline(csv_path: str, do_bootstrap: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run the full unsupervised feature selection workflow:
    1. Fill missing values
    2. Remove low-variance features
    3. Remove highly correlated features
    4. Optional bootstrap stability filtering

    Args:
        csv_path (str): Path to radiomics CSV.
        do_bootstrap (bool): Whether to perform bootstrap stability selection.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing intermediate and final DataFrames:
            - "df_low_variance": after low-variance filtering
            - "df_low_corr": after high-correlation filtering
            - "df_final": final selected features
            - "stable_features": list of final selected feature names
    """
    df = load_feature_table(csv_path)
    df = fill_missing(df)
    df_lv, kept_lv = remove_low_variance(df)
    df_corr, kept_corr = remove_high_correlation(df_lv)

    if do_bootstrap:
        stable_features = bootstrap_stable_features(df_corr)
    else:
        stable_features = df_corr.columns[1:].tolist()

    df_final = pd.concat([df_corr.iloc[:, :1], df_corr[stable_features]], axis=1)

    return {
        "df_low_variance": df_lv,
        "df_low_corr": df_corr,
        "df_final": df_final,
        "stable_features": stable_features
    }
