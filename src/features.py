import pandas as pd
from typing import List, Dict, Any

def get_feature_names() -> List[str]:
    """
    Returns the list of feature names after preprocessing (sanitized for XGBoost).
    """
    return [
        'Air temperature (K)', 
        'Process temperature (K)', 
        'Rotational speed (rpm)', 
        'Torque (Nm)', 
        'Tool wear (min)',
        'Type_H',
        'Type_L',
        'Type_M'
    ]

def get_feature_descriptions() -> Dict[str, str]:
    """
    Returns a dictionary mapping feature names to human-readable descriptions.
    """
    return {
        'Air temperature (K)': 'Air temperature in Kelvin',
        'Process temperature (K)': 'Process temperature in Kelvin',
        'Rotational speed (rpm)': 'Rotational speed of the machine',
        'Torque (Nm)': 'Torque of the machine',
        'Tool wear (min)': 'Tool wear in minutes',
        'Type_H': 'High quality product variant',
        'Type_L': 'Low quality product variant',
        'Type_M': 'Medium quality product variant'
    }

def get_feature_ranges() -> Dict[str, Dict[str, float]]:
    """
    Returns realistic min/max ranges for each raw feature.
    """
    return {
        'Air temperature [K]': {'min': 295.0, 'max': 305.0},
        'Process temperature [K]': {'min': 305.0, 'max': 315.0},
        'Rotational speed [rpm]': {'min': 1100.0, 'max': 2900.0},
        'Torque [Nm]': {'min': 3.0, 'max': 80.0},
        'Tool wear [min]': {'min': 0.0, 'max': 260.0}
    }

def prepare_single_input(product_type: str, air_temp: float, process_temp: float, 
                         rotational_speed: float, torque: float, tool_wear: float) -> pd.DataFrame:
    """
    Takes raw user inputs and returns a formatted dataframe ready for the pipeline.
    """
    data = {
        'Type': [product_type],
        'Air temperature [K]': [air_temp],
        'Process temperature [K]': [process_temp],
        'Rotational speed [rpm]': [rotational_speed],
        'Torque [Nm]': [torque],
        'Tool wear [min]': [tool_wear]
    }
    
    df = pd.DataFrame(data)
    
    # We apply the same encoding step as in preprocessing
    from src.preprocessing import encode_categorical
    df_encoded = encode_categorical(df)
    
    return df_encoded
