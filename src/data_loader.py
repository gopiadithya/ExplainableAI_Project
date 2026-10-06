import os
import pandas as pd
from typing import Dict, Any

def load_raw_data(filepath: str = "data/raw/ai4i2020.csv") -> pd.DataFrame:
    """
    Loads the AI4I 2020 dataset from the specified file path.
    
    Columns:
    - UDI: Unique identifier (1-10000)
    - Product ID: Product serial number
    - Type: Product quality variant (L, M, H)
    - Air temperature [K]: Air temperature in Kelvin
    - Process temperature [K]: Process temperature in Kelvin  
    - Rotational speed [rpm]: Rotational speed
    - Torque [Nm]: Torque
    - Tool wear [min]: Tool wear in minutes
    - Machine failure: TARGET (0/1)
    - TWF, HDF, PWF, OSF, RNF: Failure mode indicators (MUST BE EXCLUDED from features)
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading data from {filepath}: {str(e)}")

def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns a dictionary containing information about the dataset.
    """
    info = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "target_distribution": {}
    }
    
    if 'Machine failure' in df.columns:
        info["target_distribution"] = df['Machine failure'].value_counts(normalize=True).to_dict()
        
    return info

def get_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns descriptive statistics for the dataset.
    """
    return df.describe()
