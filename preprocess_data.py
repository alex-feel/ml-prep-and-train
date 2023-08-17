import argparse
import collections
import json
import os
import pickle
import warnings
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler


# Drop unnecessary columns
def drop_columns(data: pd.DataFrame, config: Dict, expect_target: bool = True) -> pd.DataFrame:
    """
    Drops the columns not specified in the provided configuration.

    Args:
        data: The input DataFrame.
        config: Configuration dictionary containing the rules for processing the DataFrame.
        expect_target: Optional; default is True. If False, the function will not expect the DataFrame to have the
                       target column specified in the configuration.

    Raises:
        ValueError: If a specified column in the configuration is not found in the DataFrame.
        ValueError: If the configuration does not contain a 'columns' key.

    Returns:
        A DataFrame with only the specified columns remaining.
    """
    if 'columns' not in config:
        raise ValueError("The configuration does not contain a 'columns' key")

    specified_columns = config['columns']

    if not expect_target and config['target_column'] in specified_columns:
        specified_columns.remove(config['target_column'])

    for column in specified_columns:
        if column not in data.columns:
            raise ValueError(
                f"Column '{column}' specified in the configuration is not found in the DataFrame")

    # Initialize a new DataFrame to avoid SettingWithCopyWarning
    data = data.copy()

    # Keep only the specified columns
    data = data[specified_columns]

    return data


# Reorder columns
def reorder_columns(data: pd.DataFrame, config: Dict, ohe_columns_dict: Optional[Dict] = None) -> pd.DataFrame:
    """
    Reorders the DataFrame columns according to the provided configuration and one-hot encoding columns dictionary.

    Args:
        data: The input DataFrame.
        config: Configuration dictionary containing the rules for processing the DataFrame.
        ohe_columns_dict: A dictionary containing the one-hot encoded columns. Each entry maps a column name to
         a dictionary, which contains the list of unique values for the column and the value that was dropped when
         the column was one-hot encoded. If this argument is not provided, no changes are made to the column order.

    Raises:
        ValueError: If the configuration does not contain a 'columns' key.

    Returns:
        A DataFrame with columns reordered according to the specified column order.
    """
    required_keys = ['columns']

    # Check if all required keys are in the configuration
    if not all(key in config for key in required_keys):
        raise ValueError(
            f'Configuration must contain the keys {required_keys}')

    specified_columns = config['columns'].copy()

    # If ohe_columns_dict is provided, modify the specified_columns accordingly
    if ohe_columns_dict is not None:
        for column, info in ohe_columns_dict.items():
            # Remove the original column name from the specified_columns
            if column in specified_columns:
                specified_columns.remove(column)

            # Add the one-hot encoded column names to the specified_columns
            for value in info['values']:
                if value != info.get('dropped'):  # Skip the dropped column
                    new_column_name = f'{column}_{value}'
                    specified_columns.append(new_column_name)

    # Initialize a new DataFrame to avoid SettingWithCopyWarning
    data = data.copy()

    # Reorder columns in DataFrame using specified_columns
    data = data.reindex(columns=specified_columns)

    return data


# Convert to number
def convert_to_number(data: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Convert specified DataFrame columns to int or float.

    Args:
        data: The input DataFrame.
        config: Configuration dictionary containing rules for converting to numbers.

    Raises:
        ValueError: If the specified column is not found in the data.
        ValueError: If the type to convert to is not 'int' or 'float'.
        ValueError: If each column configuration does not contain the keys ['column', 'to_type'].

    Returns:
        The transformed DataFrame.
    """
    rules = config['converting_to_number']

    required_keys = ['column', 'to_type']

    for rule in rules:
        # Check if all required keys are in the configuration
        if not all(key in rule for key in required_keys):
            raise ValueError(
                f'Each column configuration must contain the keys {required_keys}')

        column_to_convert = rule['column']
        to_type = rule['to_type']

        if column_to_convert not in data.columns:
            raise ValueError(f"Column '{column_to_convert}' not found in data")

        if to_type not in ['int', 'float']:
            raise ValueError(
                f"Type '{to_type}' not recognized. Only 'int' and 'float' are supported.")

        # Initialize a new DataFrame to avoid SettingWithCopyWarning
        data = data.copy()

        # Convert column to the desired type
        data[column_to_convert] = data[column_to_convert].astype(to_type)

    return data


# Apply One-Hot Encoding (OHE)
def apply_one_hot_encoding(data: pd.DataFrame, config: Dict, full_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply One-Hot Encoding (OHE) to the DataFrame.

    Args:
        data: The input DataFrame.
        config: Configuration dictionary containing rules for applying one-hot encoding.
        full_data: The DataFrame before splitting to the training and testing sets.

    Raises:
        ValueError: If the specified column is not found in the data.
        ValueError: If each column configuration does not contain the keys ['column', 'drop_first'].

    Returns:
        Tuple containing the transformed DataFrame and a dictionary with the OHE columns.
    """
    rules = config['one_hot_encoding']
    ohe_columns_dict = {}
    most_frequent_value = ''

    required_keys = ['column', 'drop_first']

    for rule in rules:
        # Check if all required keys are in the configuration
        if not all(key in rule for key in required_keys):
            raise ValueError(
                f'Each column configuration must contain the keys {required_keys}')

        column_to_apply = rule['column']
        drop_first = rule['drop_first']

        if not column_to_apply:
            return data, ohe_columns_dict

        if column_to_apply not in full_data.columns:
            raise ValueError(
                f"Column '{column_to_apply}' not found in full_data")

        unique_values = full_data[column_to_apply].unique()

        if drop_first:
            most_frequent_value = full_data[column_to_apply].value_counts(
            ).idxmax()
            ohe_columns_dict[column_to_apply] = {
                'values': unique_values.tolist(), 'dropped': most_frequent_value}
        else:
            ohe_columns_dict[column_to_apply] = {
                'values': unique_values.tolist()}

        # Initialize a new DataFrame to avoid SettingWithCopyWarning
        data = data.copy()

        # Initialize all OHE columns with 0
        for value in unique_values:
            data[f'{column_to_apply}_{value}'] = 0

        # Manual One-Hot Encoding based on unique_values
        for value in data[column_to_apply].unique():
            data.loc[data[column_to_apply] == value,
                     f'{column_to_apply}_{value}'] = 1

        # Drop the first column if necessary
        if drop_first:
            data = data.drop(column_to_apply + '_' +
                             str(most_frequent_value), axis=1)

        # Removing original columns from the data
        data.drop(column_to_apply, axis=1, inplace=True)

    return data, ohe_columns_dict


# Apply One-Hot Encoding (OHE) using dictionary
def apply_one_hot_encoding_using_dict(data: pd.DataFrame, config: Dict, ohe_columns_dict: Dict) -> pd.DataFrame:
    """
    Apply One-Hot Encoding (OHE) to the DataFrame using a dictionary.

    Args:
        data: The input DataFrame.
        config: Configuration dictionary containing rules for applying one-hot encoding.
        ohe_columns_dict: Dictionary containing the OHE columns from the dictionary.

    Raises:
        ValueError: If the specified column is not found in the data.
        ValueError: If each column configuration does not contain the keys ['column', 'drop_first'].

    Returns:
        DataFrame with one-hot encoding applied using the specified dictionary.
    """
    rules = config['one_hot_encoding']

    required_keys = ['column', 'drop_first']

    for rule in rules:
        # Check if all required keys are in the configuration
        if not all(key in rule for key in required_keys):
            raise ValueError(
                f'Each column configuration must contain the keys {required_keys}')

        column_to_apply = rule['column']
        drop_first = rule['drop_first']

        if not column_to_apply:
            return data

        if column_to_apply not in data.columns:
            raise ValueError(f"Column '{column_to_apply}' not found in data")

        unique_values = ohe_columns_dict[column_to_apply]['values']
        dropped_column = ohe_columns_dict[column_to_apply].get('dropped')

        # Initialize a new DataFrame to avoid SettingWithCopyWarning
        data = data.copy()

        # Initialize all OHE columns with 0
        for value in unique_values:
            data[f'{column_to_apply}_{value}'] = 0

        # If the value exists in the data, set the corresponding OHE column to 1
        for value in data[column_to_apply].unique():
            data.loc[data[column_to_apply] == value,
                     f'{column_to_apply}_{value}'] = 1

        # Drop the first column if necessary
        if drop_first:
            data = data.drop(columns=[f'{column_to_apply}_{dropped_column}'])

        # Removing original columns from the data
        data.drop(column_to_apply, axis=1, inplace=True)

    return data


# Fill missing numeric values with zeros
def fill_missing_numeric_values_with_zeros(data: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Fill missing numeric values in the DataFrame with zeros.

    Args:
        data: The input DataFrame.
        config: Configuration dictionary containing rules for filling missing values.

    Raises:
        ValueError: If the specified column is not found in the data.

    Returns:
        DataFrame with missing numeric values filled with zeros.
    """
    columns_to_fill = config['filling_missing_numeric_values_with_zeroes']

    for column in columns_to_fill:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")

        # Initialize a new DataFrame to avoid SettingWithCopyWarning
        data = data.copy()

        data[column] = data[column].fillna(0)

    return data


# Fill missing numeric values using specified statistical methods
def fill_missing_numeric_values_with_stats_method(data: pd.DataFrame, config: Dict, config_key: str) \
        -> Tuple[pd.DataFrame, Dict]:
    """
    Fill missing numeric values in the DataFrame using specified statistical methods and save the statistics.

    Args:
        data: The input DataFrame.
        config: Configuration dictionary containing rules for filling missing values.
        config_key: Key to access the specific configuration rules in the config dictionary.

    Raises:
        ValueError: If the specified column is not found in the data, if an invalid fill_mode is provided,
            or if the mode cannot be calculated for a column.
        ValueError: If each column configuration does not contain the keys ['column', 'fill_missing_values_method'].

    Returns:
        Tuple containing the DataFrame with missing numeric values filled and a dictionary with the used statistics.
    """
    columns_to_fill = config[config_key]
    num_stats_dict = {}

    required_keys = ['column', 'fill_missing_values_method']

    for column_config in columns_to_fill:
        if not column_config:
            return data, num_stats_dict

        # Check if all required keys are in the configuration
        if not all(key in column_config for key in required_keys):
            raise ValueError(
                f'Each column configuration must contain the keys {required_keys}')

        column_to_fill = column_config['column']
        fill_mode = column_config['fill_missing_values_method']

        if column_to_fill not in data.columns:
            raise ValueError(f"Column '{column_to_fill}' not found in data")

        if not fill_mode:
            continue

        # Initialize a new DataFrame to avoid SettingWithCopyWarning
        data = data.copy()

        if fill_mode == 'mode':
            mode_values = data[column_to_fill].mode()
            if mode_values.empty:
                raise ValueError(
                    f"Cannot calculate mode for column '{column_to_fill}'")
            fill_value = mode_values[0]
        elif fill_mode == 'median':
            fill_value = data[column_to_fill].median()
        elif fill_mode == 'mean':
            fill_value = data[column_to_fill].mean()
        else:
            raise ValueError(
                "fill_mode must be one of 'mode', 'median', or 'mean'")

        data[column_to_fill] = data[column_to_fill].fillna(fill_value)

        # Saving statistics
        num_stats_dict[column_to_fill] = fill_value

    return data, num_stats_dict


# Fill missing numeric values using dictionary
def fill_missing_numeric_values_with_stats_method_using_dict(data: pd.DataFrame, config: Dict, config_key: str,
                                                             num_stats_dict: Dict) -> pd.DataFrame:
    """
    Fill missing numeric values in the DataFrame using statistics from the dictionary.

    Args:
        data: The input DataFrame.
        config: Configuration dictionary containing rules for filling missing values.
        config_key: Key to access the specific configuration rules in the config dictionary.
        num_stats_dict: Dictionary with the statistics used for filling missing values.

    Raises:
        ValueError: If the specified column is not found in the data.
        ValueError: If the required keys are not present in the config dictionary.

    Returns:
        DataFrame with missing numeric values filled according to the statistics in the dictionary.
    """
    columns_to_fill = config[config_key]

    required_keys = ['column']

    for column_config in columns_to_fill:
        if not column_config:
            return data

        # Check if all required keys are in the configuration
        if not all(key in column_config for key in required_keys):
            raise ValueError(
                f'Each column configuration must contain the keys {required_keys}')

        column_to_fill = column_config['column']

        if column_to_fill not in data.columns:
            raise ValueError(f"Column '{column_to_fill}' not found in data")

        fill_value = num_stats_dict[column_to_fill]

        # Initialize a new DataFrame to avoid SettingWithCopyWarning
        data = data.copy()

        # Filling missing values using the statistics from the dictionary
        data[column_to_fill] = data[column_to_fill].fillna(fill_value)

    return data


# Fill missing text values using specified methods
def fill_missing_text_values_with_method(data: pd.DataFrame, config: Dict, config_key: str) \
        -> Tuple[pd.DataFrame, Dict]:
    """
    Fill missing text values in the DataFrame using specified methods and save the statistics.

    Args:
        data: The input DataFrame.
        config: Configuration dictionary containing rules for filling missing values.
        config_key: Key to access the specific configuration rules in the config dictionary.

    Raises:
        ValueError: If the specified column is not found in the data, if an invalid fill_method is provided,
            if a placeholder is required but not provided, or if the mode cannot be calculated for a column.
        ValueError: If the required keys are not present in the config dictionary.

    Returns:
        Tuple containing the DataFrame with missing text values filled and a dictionary with the used statistics.
    """
    columns_to_fill = config[config_key]
    text_stats_dict = {}

    required_keys = ['column', 'fill_missing_values_method']

    for column_config in columns_to_fill:
        if not column_config:
            return data, text_stats_dict

        # Check if all required keys are in the configuration
        if not all(key in column_config for key in required_keys):
            raise ValueError(
                f'Each column configuration must contain the keys {required_keys}')

        column_to_fill = column_config['column']
        fill_method = column_config['fill_missing_values_method']
        placeholder = column_config.get('placeholder', '')

        if column_to_fill not in data.columns:
            raise ValueError(f"Column '{column_to_fill}' not found in data")

        if not fill_method:
            continue

        if fill_method == 'placeholder' and (placeholder is None or placeholder == ''):
            raise ValueError(
                f"'placeholder' must be specified and not an empty string for column '{column_to_fill}' when"
                f"'fill_missing_values_method' is set to 'placeholder'")

        # Initialize a new DataFrame to avoid SettingWithCopyWarning
        data = data.copy()

        if fill_method == 'mode':
            mode_values = data[column_to_fill].mode()
            if mode_values.empty:
                raise ValueError(
                    f"Cannot calculate mode for column '{column_to_fill}'")
            fill_value = mode_values[0]
        elif fill_method == 'placeholder':
            fill_value = placeholder
        else:
            raise ValueError(
                "fill_method must be one of 'mode' or 'placeholder'")

        data[column_to_fill] = data[column_to_fill].fillna(fill_value)

        # Saving statistics
        text_stats_dict[column_to_fill] = fill_value

    return data, text_stats_dict


# Fill missing text values using dictionary
def fill_missing_text_values_with_method_using_dict(data: pd.DataFrame, config: Dict, config_key: str,
                                                    text_stats_dict: Dict) -> pd.DataFrame:
    """
    Fill missing text values in the DataFrame using statistics from the dictionary.

    Args:
        data: The input DataFrame.
        config: Configuration dictionary containing rules for filling missing values.
        config_key: Key to access the specific configuration rules in the config dictionary.
        text_stats_dict: Dictionary with the statistics used for filling missing values.

    Raises:
        ValueError: If the specified column is not found in the data.
        ValueError: If any column configuration doesn't contain all required keys.

    Returns:
        DataFrame with missing text values filled according to the statistics in the dictionary.
    """
    columns_to_fill = config[config_key]

    required_keys = ['column']

    for column_config in columns_to_fill:
        if not column_config:
            return data

        # Check if all required keys are in the configuration
        if not all(key in column_config for key in required_keys):
            raise ValueError(
                f'Each column configuration must contain the keys {required_keys}')

        column_to_fill = column_config['column']

        # Initialize a new DataFrame to avoid SettingWithCopyWarning
        data = data.copy()

        if column_to_fill not in data.columns:
            raise ValueError(f"Column '{column_to_fill}' not found in data")

        fill_value = text_stats_dict[column_to_fill]

        # Filling missing values using the statistics from the dictionary
        data[column_to_fill] = data[column_to_fill].fillna(fill_value)

    return data


# Remove outliers
def remove_outliers(data: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Remove outliers from the DataFrame and save the thresholds.

    Args:
        data: The input DataFrame.
        config: Configuration dictionary containing rules for removing outliers.

    Raises:
        ValueError: If an invalid search_method is provided.
        ValueError: If any column configuration doesn't contain all required keys.

    Returns:
        Tuple containing DataFrame with outliers removed and a dictionary with the used thresholds.
    """
    removing_outliers_config = config['removing_outliers']
    thresholds_dict = {}

    required_keys = ['column', 'search_method']

    for config_info in removing_outliers_config:
        if not config_info:
            return data, thresholds_dict

        # Check if all required keys are in the configuration
        if not all(key in config_info for key in required_keys):
            raise ValueError(
                f'Each column configuration must contain the keys {required_keys}')

        column = config_info['column']
        search_method = config_info['search_method']

        # Initialize a new DataFrame to avoid SettingWithCopyWarning
        data = data.copy()

        if search_method == 'z_score':
            threshold = config_info.get('threshold', 3)
            z_scores = (data[column] - data[column].mean()) / \
                data[column].std()
            data = data[np.abs(z_scores) < threshold]
            thresholds = {'lower': -threshold, 'upper': threshold}

        elif search_method in ['iqr', 'tukey_fences']:
            k = config_info.get('k', 1.5 if search_method == 'iqr' else 3.0)
            q1 = data[column].quantile(0.25)
            q3 = data[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - k * iqr
            upper_bound = q3 + k * iqr
            data = data[(data[column] > lower_bound) &
                        (data[column] < upper_bound)]
            thresholds = {'lower': lower_bound, 'upper': upper_bound}

        else:
            raise ValueError(
                "search_method must be one of 'z_score', 'iqr', or 'tukey_fences'")

        # Saving thresholds
        thresholds_dict[column] = {
            'bounds': thresholds,
            'method': search_method
        }

    return data, thresholds_dict


# Remove outliers using dictionary
def remove_outliers_using_dict(data: pd.DataFrame, config: Dict, thresholds_dict: Dict) -> pd.DataFrame:
    """
    Remove outliers from the test set DataFrame using thresholds from the dictionary.

    Args:
        data: The input DataFrame.
        config: Configuration dictionary containing rules for removing outliers.
        thresholds_dict: Dictionary with the thresholds used for removing outliers.

    Raises:
        ValueError: If any column configuration doesn't contain all required keys.

    Returns:
        DataFrame with outliers removed according to the thresholds in the dictionary.
    """
    removing_outliers_config = config['removing_outliers']

    required_keys = ['column']

    for config_info in removing_outliers_config:
        if not config_info:
            return data

        # Check if all required keys are in the configuration
        if not all(key in config_info for key in required_keys):
            raise ValueError(
                f'Each column configuration must contain the keys {required_keys}')

        column = config_info['column']
        lower_bound, upper_bound = thresholds_dict[column]['bounds']

        # Initialize a new DataFrame to avoid SettingWithCopyWarning
        data = data.copy()

        # Removing outliers using the thresholds from the dictionary
        data = data[(data[column] > lower_bound) &
                    (data[column] < upper_bound)]

    return data


# Replace outliers
def replace_outliers(data: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Replace outliers in the DataFrame and save the replacement values.

    Args:
        data: The input DataFrame.
        config: Configuration dictionary containing rules for replacing outliers.

    Raises:
        ValueError: If an invalid search_method or replacement_method is provided.
        ValueError: If any column configuration doesn't contain all required keys.

    Returns:
        Tuple containing DataFrame with outliers replaced and a dictionary with the used replacement values.
    """
    replacing_outliers_config = config['replacing_outliers']
    replacement_values_dict = {}

    required_keys = ['column', 'search_method', 'replacement_method']

    for config_info in replacing_outliers_config:
        if not config_info:
            return data, replacement_values_dict

        # Check if all required keys are in the configuration
        if not all(key in config_info for key in required_keys):
            raise ValueError(
                f'Each column configuration must contain the keys {required_keys}')

        column = config_info['column']
        search_method = config_info['search_method']
        replacement_method = config_info['replacement_method']

        # Initialize a new DataFrame to avoid SettingWithCopyWarning
        data = data.copy()

        if search_method == 'z_score':
            threshold = config_info.get('threshold', 3)
            mean = data[column].mean()
            std = data[column].std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            outliers = (data[column] < lower_bound) | (
                data[column] > upper_bound)

        elif search_method in ['iqr', 'tukey_fences']:
            k = config_info.get('k', 1.5 if search_method == 'iqr' else 3.0)
            q1 = data[column].quantile(0.25)
            q3 = data[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - k * iqr
            upper_bound = q3 + k * iqr
            outliers = (data[column] < lower_bound) | (
                data[column] > upper_bound)

        else:
            raise ValueError(
                "search_method must be one of 'z_score', 'iqr', or 'tukey_fences'")

        if replacement_method in ['mean', 'median']:
            replacement_value = getattr(data[column], replacement_method)()
        elif replacement_method == 'mode':
            replacement_value = getattr(
                data[column], replacement_method)().item()
        elif replacement_method == 'quantile_25':
            replacement_value = data[column].quantile(0.25)
        elif replacement_method == 'quantile_75':
            replacement_value = data[column].quantile(0.75)
        elif replacement_method in ['lower_bound', 'upper_bound']:
            replacement_value = locals()[replacement_method]
        else:
            raise ValueError("replacement_method must be one of 'mean', 'median', 'mode', 'quantile_25', "
                             "'quantile_75', 'lower_bound', or 'upper_bound'")

        data.loc[outliers, column] = replacement_value

        # Saving replacement values
        replacement_values_dict[column] = {
            'value': replacement_value,
            'bounds': (lower_bound, upper_bound),
            'method': search_method
        }

    return data, replacement_values_dict


# Replace outliers using dictionary
def replace_outliers_using_dict(data: pd.DataFrame, config: Dict, replacement_values_dict: Dict) -> pd.DataFrame:
    """
    Replace outliers in the DataFrame using replacement values from the dictionary.

    Args:
        data: The input DataFrame.
        config: Configuration dictionary containing rules for replacing outliers.
        replacement_values_dict: Dictionary with the replacement values used for replacing outliers.

    Raises:
        ValueError: If any column configuration doesn't contain all required keys.

    Returns:
        DataFrame with outliers replaced according to the replacement values in the dictionary.
    """
    replacing_outliers_config = config['replacing_outliers']

    required_keys = ['column']

    for config_info in replacing_outliers_config:
        if not config_info:
            return data

        # Check if all required keys are in the configuration
        if not all(key in config_info for key in required_keys):
            raise ValueError(
                f'Each column configuration must contain the keys {required_keys}')

        column = config_info['column']
        replacement_value = replacement_values_dict[column]['value']
        lower_bound, upper_bound = replacement_values_dict[column]['bounds']

        # Initialize a new DataFrame to avoid SettingWithCopyWarning
        data = data.copy()

        outliers = (data[column] < lower_bound) | (data[column] > upper_bound)

        # Replacing outliers using the replacement values from the dictionary
        data.loc[outliers, column] = replacement_value

    return data


# Transform data
def transform_data(data: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Transform data in the DataFrame and save the transformation parameters.

    Args:
        data: The input DataFrame.
        config: Configuration dictionary containing rules for data transformation.

    Raises:
        ValueError: If an invalid transformation_method is provided or if power is not specified for the 'power' method.
        ValueError: If any column configuration doesn't contain all required keys.

    Returns:
        Tuple containing DataFrame with data transformed and a dictionary with the used transformation parameters.
    """
    transformation_config = config['transformation']
    transformation_params_dict = {}

    required_keys = ['column', 'transformation_method']

    for config_info in transformation_config:
        if not config_info:
            return data, transformation_params_dict

        # Check if all required keys are in the configuration
        if not all(key in config_info for key in required_keys):
            raise ValueError(
                f'Each column configuration must contain the keys {required_keys}')

        column = config_info['column']
        transformation_method = config_info['transformation_method']
        power = config_info.get('power')

        # Initialize a new DataFrame to avoid SettingWithCopyWarning
        data = data.copy()

        if transformation_method == 'log':
            data[column] = np.log(data[column] + 1)  # Adding 1 to avoid log(0)
            transformation_params_dict[column] = {
                'transformation_method': 'log'}

        elif transformation_method == 'sqrt':
            data[column] = np.sqrt(data[column])
            transformation_params_dict[column] = {
                'transformation_method': 'sqrt'}

        elif transformation_method == 'power':
            if power is None:
                raise ValueError(
                    "Power must be specified for 'power' transformation method")
            data[column] = data[column] ** power
            transformation_params_dict[column] = {
                'transformation_method': 'power', 'power': power}

        else:
            raise ValueError(
                "transformation_method must be one of 'log', 'sqrt', or 'power'")

    return data, transformation_params_dict


# Transform data using dictionary
def transform_data_using_dict(data: pd.DataFrame, config: Dict, transformation_params_dict: Dict) -> pd.DataFrame:
    """
    Transform data in the DataFrame using transformation parameters from the dictionary.

    Args:
        data: The input DataFrame.
        config: Configuration dictionary containing rules for data transformation.
        transformation_params_dict: Dictionary with the transformation parameters used for data transformation.

    Raises:
        ValueError: If an invalid transformation_method is provided or if power is not specified for the 'power' method.
        ValueError: If any column configuration doesn't contain all required keys.

    Returns:
        DataFrame with data transformed according to the transformation parameters in the dictionary.
    """
    transformation_config = config['transformation']

    required_keys = ['column']

    for config_info in transformation_config:
        if not config_info:
            return data

        # Check if all required keys are in the configuration
        if not all(key in config_info for key in required_keys):
            raise ValueError(
                f'Each column configuration must contain the keys {required_keys}')

        column = config_info['column']
        transformation_params = transformation_params_dict[column]
        transformation_method = transformation_params['transformation_method']
        power = transformation_params.get('power')

        # Initialize a new DataFrame to avoid SettingWithCopyWarning
        data = data.copy()

        if transformation_method == 'log':
            data[column] = np.log(data[column] + 1)

        elif transformation_method == 'sqrt':
            data[column] = np.sqrt(data[column])

        elif transformation_method == 'power':
            if power is None:
                raise ValueError(
                    "Power must be specified for 'power' transformation method")
            data[column] = data[column] ** power

        else:
            raise ValueError(
                "transformation_method must be one of 'log', 'sqrt', or 'power'")

    return data


# Apply Ordinal Encoding
def apply_ordinal_encoding(data: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Apply Ordinal Encoding to the DataFrame.

    Args:
        data: The input DataFrame.
        config: Configuration dictionary containing rules for applying ordinal encoding.

    Raises:
        ValueError: If the specified column is not found in the data or if some values in the column
            are not present in the provided ordinal mapping.
        ValueError: If the same numeric encoding is assigned to more than one category.
        ValueError: If any column configuration doesn't contain all required keys.

    Returns:
        DataFrame with ordinal encoding applied according to the specified rules.
    """
    ordinal_encoding_config = config['ordinal_encoding']

    required_keys = ['column', 'mapping']

    for encoding_info in ordinal_encoding_config:
        if not encoding_info:
            return data

        # Check if all required keys are in the configuration
        if not all(key in encoding_info for key in required_keys):
            raise ValueError(
                f'Each column configuration must contain the keys {required_keys}')

        column = encoding_info['column']
        ordinal_mapping = encoding_info['mapping']

        # Convert the keys of the ordinal_mapping to match the dtype of the column
        column_dtype = data[column].dtype
        ordinal_mapping = {column_dtype.type(
            k): v for k, v in ordinal_mapping.items()}

        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")

        if not set(data[column].unique()).issubset(set(ordinal_mapping.keys())):
            raise ValueError(
                f"Some values in '{column}' are not present in the provided ordinal mapping")

        # Check for duplicate encoding values
        counter = collections.Counter(ordinal_mapping.values())
        duplicates = [value for value, count in counter.items() if count > 1]
        if duplicates:
            raise ValueError(
                f"The same numeric encoding {duplicates} is assigned to more than one category in '{column}'. "
                f"This will result in data loss as these categories will be merged after encoding. "
                f"Please review the encoding rules for this column.")

        # Check for category encoded as 0
        if 0 in ordinal_mapping.values():
            warnings.warn(
                f"A category in column '{column}' is encoded as 0. "
                f"If 0 does not signify the absence of something, it's generally better to replace it with another "
                f"number to avoid potential confusion or issues in subsequent data analysis. "
                f"Please review the encoding rules for this column.")

        mapped_values = sorted(ordinal_mapping.values())
        diff_values = np.diff(mapped_values)
        if any(diff_values != 1):
            missing_numbers = [mapped_values[i] + j for i,
                               gap in enumerate(diff_values) for j in range(1, gap)]
            warnings.warn(f"The ordinal mapping for '{column}' is missing the following numbers: {missing_numbers}. "
                          f"This may result in loss of information during encoding. "
                          f"Please review the encoding rules for this column.")

        # Initialize a new DataFrame to avoid SettingWithCopyWarning
        data = data.copy()

        data[column] = data[column].map(ordinal_mapping)

    return data


# Get min and max values
def get_min_max_values(data: pd.DataFrame, column_config: Dict) -> Tuple[float, float]:
    """
    Check if the column in DataFrame is numeric and if so, return its minimum and maximum values.

    Args:
        data: The input DataFrame before splitting into training and test sets, it is important that the minimum
            and maximum values are calculated on the entire dataset.
        column_config: Configuration dictionary for a column, containing the column name.

    Raises:
        ValueError: If the specified column is not numeric.

    Returns:
        Minimum and maximum value of the specified column.
    """
    column = column_config['column']

    if data[column].dtype not in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
        raise ValueError(f"Column '{column}' is not numeric")

    min_value = data[column].min()
    max_value = data[column].max()

    return min_value, max_value


# Check if bins cover all values
def validate_manual_bins(data: pd.DataFrame, column_config: Dict):
    """
    Checks if the specified bins cover all existing values in the DataFrame for the current column and
    also checks if there are any bins that do not contain any values.
    This function applies only for the 'manual' binning method.

    Args:
        data: The DataFrame before splitting into training and test sets.
        column_config: Configuration dictionary for a column, containing the column name and bins.

    Raises:
        ValueError: If the minimum value in the column is less than the minimum bin,
            or if the maximum value in the column is more than the maximum bin,
            or if there are bins that do not contain any values.
    """
    column = column_config['column']
    binning_method = column_config.get(
        'binning_method', 'equal_number_of_bins')

    if binning_method != 'manual':
        return

    bins = column_config.get('bins', [])
    if not bins:
        raise ValueError(
            f"bins must be provided for 'manual' binning method for column '{column}'.")

    min_value = data[column].min()
    if min_value < min(bins):
        raise ValueError(
            f"Minimum value {min_value} in column '{column}' is less than the minimum bin")

    max_value = data[column].max()
    if max_value > max(bins):
        raise ValueError(
            f"Maximum value {max_value} in column '{column}' is more than the maximum bin")

    # Create bins ranges
    bins_ranges = [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]

    # Check if any bin does not contain any values
    empty_bins = [bins_range for bins_range in bins_ranges if data[column].between(
        *bins_range).sum() == 0]
    if empty_bins:
        raise ValueError(
            f"The following bins for column '{column}' do not contain any values: {empty_bins}")


# Calculate value count
def calculate_value_count(data: pd.DataFrame, column: str, lower_percent: float = 0.02,
                          upper_percent: float = 0.1, min_count: int = 20):
    """
    Calculates the number of values corresponding to the square root of the total number of values in the column.
    If the square root is less than the lower_percent or more than the upper_percent of the total values,
    returns the number corresponding to these percentages along with these boundaries.

    Args:
        data: The input DataFrame.
        column: The name of the column.
        lower_percent: The lower percentage limit.
        upper_percent: The upper percentage limit.
        min_count: The minimum number of values in a bin.

    Returns:
        Tuple of three values: The calculated number of values, the lower boundary, and the upper boundary.
    """
    # Get the total number of values in the column
    total_values = data[column].count()

    # Calculate the square root of total number of values
    sqrt_value = round(np.sqrt(total_values))

    # Calculate the boundaries
    lower_boundary = round(total_values * lower_percent)
    upper_boundary = round(total_values * upper_percent)

    # Check if square root value is less than lower_percent or more than upper_percent of total values
    if sqrt_value < lower_boundary:
        count = lower_boundary
    elif sqrt_value > upper_boundary:
        count = upper_boundary
    else:
        count = sqrt_value

    # If the count is less than the minimum count, replace it with the minimum count
    if count < min_count:
        count = min_count

    return count, lower_boundary, upper_boundary


# Find optimal quantiles
def find_optimal_quantiles(data: pd.DataFrame, column: str):
    """
    Finds the maximum number of quantiles for the given data such that each bin has approximately equal number
    of observations, but not less than 'min_bins' observations.

    Args:
        data (pd.DataFrame): The input DataFrame.
        column (str): The column to bin.

    Returns:
        int: The maximum number of quantiles.
    """
    n = len(data)
    min_bins, _, _ = calculate_value_count(data, column)
    max_quantiles = n // min_bins  # Maximum number of quantiles

    for q in range(max_quantiles, 0, -1):
        bins, _ = pd.qcut(data[column], q=q, retbins=True, duplicates='drop')
        bin_counts = bins.value_counts()
        if bin_counts.min() >= min_bins:
            return q
    return 1  # Return 1 if no suitable number of quantiles found


# Check bin counts
def check_bin_counts(data: pd.DataFrame, column: str):
    """
    Checks the number of values in each bin of a column and prints a warning if it's outside the expected range.

    Args:
        data: The input DataFrame.
        column: The name of the column.
    """
    # Get the number of values in each bin and their expected range
    _, lower_boundary, upper_boundary = calculate_value_count(data, column)
    bin_counts = data[column].value_counts().sort_index()

    # Print warnings for bins outside the expected range
    first_warning = True
    for bin_number, count in bin_counts.items():
        if count < lower_boundary or count > upper_boundary:
            if first_warning:
                print(f"\nWarnings for column '{column}':")
                first_warning = False
            print(f'\tThe number of values in bin {bin_number} ({count}) is outside the expected range '
                  f'({lower_boundary}, {upper_boundary})')


# Apply binning
def apply_binning(data: pd.DataFrame, column_config: Dict, min_value: float = None, max_value: float = None) \
        -> Tuple[pd.DataFrame, Any]:
    """
    Apply binning to the DataFrame column based on the provided configuration.

    Args:
        data (pd.DataFrame): The input DataFrame.
        column_config (Dict): Configuration dictionary for a column containing rules for applying binning.
        min_value (float, optional): The minimum value for the binning range. Defaults to None.
        max_value (float, optional): The maximum value for the binning range. Defaults to None.

    Raises:
        ValueError: If the binning method is invalid or if min_value and max_value are not provided for
            'equal_number_of_bins' binning method.
        ValueError: If bins are not provided for 'manual' binning method.
        ValueError: If all bins are not unique for 'manual' binning method.
        ValueError: If bins are not in increasing order for 'manual' binning method.
        ValueError: If bins are not a positive integer or 'max' for 'equal_size' binning method.
        ValueError: If bins are not a positive integer for 'equal_number_of_bins' binning method.

    Returns:
        Tuple[DataFrame, Any]: DataFrame with binning applied to the specified column, and the bin boundaries.
    """
    column = column_config['column']
    binning_method = column_config.get(
        'binning_method', 'equal_number_of_bins')

    # Initialize a new DataFrame to avoid SettingWithCopyWarning
    data = data.copy()

    if binning_method == 'equal_size':
        quantiles = column_config.get('quantiles')
        if isinstance(quantiles, str):
            if quantiles != 'max':
                raise ValueError(
                    f"quantiles must be a positive integer or 'max' for 'equal_size' binning method for column"
                    f" '{column}'.")
            quantiles = find_optimal_quantiles(data, column)
        elif isinstance(quantiles, int):
            if quantiles <= 0:
                raise ValueError(
                    f"quantiles must be a positive integer for 'equal_size' binning method for column '{column}'.")
        else:
            raise ValueError(
                f"quantiles must be a positive integer or 'max' for 'equal_size' binning method for column '{column}'.")

        if min_value is None or max_value is None:
            raise ValueError(
                "min_value and max_value must be provided for 'equal_size' binning method.")

        data[column], bin_borders = pd.qcut(
            data[column], q=quantiles, retbins=True, labels=False, duplicates='drop')

        # Adjust bin borders based on min_value and max_value
        bin_borders[0] = min(bin_borders[0], min_value)
        bin_borders[-1] = max(bin_borders[-1], max_value)

    elif binning_method == 'manual':
        # Apply manual binning based on the provided bins
        bins = column_config.get('bins', [])

        if not bins:
            raise ValueError(
                f"bins must be provided for 'manual' binning method for column '{column}'.")
        if len(set(bins)) != len(bins):
            raise ValueError(
                f"All bins must be unique for 'manual' binning method for column '{column}'.")
        if sorted(bins) != bins:
            raise ValueError(
                f"bins must be in increasing order for 'manual' binning method for column '{column}'.")

        data[column], bin_borders = pd.cut(
            data[column], bins=bins, retbins=True, labels=False, include_lowest=True)

        # Check the number of values in each bin
        check_bin_counts(data, column)

    elif binning_method == 'equal_number_of_bins':
        bins = column_config.get('bins')
        if not isinstance(bins, int) or bins <= 0:
            raise ValueError(
                f"bins must be a positive integer for 'equal_number_of_bins' binning method for column '{column}'.")
        if min_value is None or max_value is None:
            raise ValueError(
                "min_value and max_value must be provided for 'equal_number_of_bins' binning method.")

        bin_borders = np.linspace(min_value, max_value, bins + 1)
        data[column], bin_borders = pd.cut(
            data[column], bins=bin_borders, retbins=True, labels=False, include_lowest=True)

        # Check the number of values in each bin
        check_bin_counts(data, column)

    else:
        raise ValueError(
            "Invalid method. Choose 'equal_size', 'equal_number_of_bins', or 'manual'")

    # Check if variable has become constant after binning
    if len(data[column].unique()) == 1:
        raise ValueError(
            f"The variable '{column}' has become constant after binning, which may lead to errors in further analysis.")

    return data, bin_borders.tolist()


# Check category counts
def check_category_counts_and_warn(data: pd.DataFrame, column: str, threshold: int = 20):
    """
    Checks the count of each unique value in a column and provides warnings if the count is below a certain threshold.

    Args:
        data: The input DataFrame.
        column: The name of the column.
        threshold: The minimum acceptable count for each unique value in the column.

    Returns:
        None. Prints a warning if any unique value count is below the threshold.
    """
    value_counts = data[column].value_counts()

    low_counts = value_counts[value_counts < threshold].sort_index()
    num_bins = data[column].nunique()

    if not low_counts.empty:
        if pd.api.types.is_numeric_dtype(data[column]):
            print(f"Warning: The binned column '{column}' contains {num_bins} bins, with some bins having less "
                  f"than {threshold} instances. "
                  f"These bins are: \n{low_counts}\n"
                  f"Applying Target Encoding to this column might lead to overfitting.")
        else:
            print(f"Warning: The categorical column '{column}' contains {num_bins} categories, with some categories "
                  f"having less than {threshold} instances."
                  f"These categories are: \n{low_counts}\n"
                  f"Applying Target Encoding to this column might lead to overfitting.")


# Apply Target Encoding
def apply_target_encoding(data: pd.DataFrame, config: Dict, full_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply Target Encoding to the DataFrame.

    Args:
        data: The input DataFrame.
        config: Configuration dictionary containing rules for applying target encoding.
        full_data: The DataFrame before splitting to the training and testing sets to pass to get_min_max_values().

    Raises:
        ValueError: If the specified column or target column is not found in the data, if an invalid binning method is
            specified for continuous variables, or if an invalid encoding method is specified.
        ValueError: If any column configuration doesn't contain all required keys.

    Returns:
        Tuple containing the transformed DataFrame and a dictionary with the target encoding statistics.
    """
    target_encoding_config = config['target_encoding']
    target_encoding_dict = {}

    if not target_encoding_config or not any(target_encoding_config):
        return data, target_encoding_dict

    required_keys = ['column', 'type', 'target_column', 'encoding_method']

    for column_config in target_encoding_config:
        if not column_config:
            continue

        # Check if all required keys are in the configuration
        if not all(key in column_config for key in required_keys):
            raise ValueError(
                f'Each column configuration must contain the keys {required_keys}')

        column = column_config['column']
        column_type = column_config['type']
        target_column = column_config['target_column']
        encoding_method = column_config['encoding_method']

        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")

        if target_column not in data.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in data")

        target_encoding_dict[column] = {}

        # Initialize a new DataFrame to avoid SettingWithCopyWarning
        data = data.copy()

        # Apply binning for continuous variables
        if column_type == 'continuous':
            binning_method = column_config.get(
                'binning_method', 'equal_number_of_bins')

            if binning_method in ('equal_size', 'manual'):
                validate_manual_bins(full_data, column_config)
                min_value, max_value = get_min_max_values(
                    full_data, column_config)
                data, bins = apply_binning(
                    data, column_config, min_value, max_value)
                target_encoding_dict[column]['bins'] = bins

            elif binning_method == 'equal_number_of_bins':
                min_value, max_value = get_min_max_values(
                    full_data, column_config)
                data, bins = apply_binning(
                    data, column_config, min_value, max_value)
                target_encoding_dict[column]['bins'] = bins

            else:
                raise ValueError(
                    "Invalid method. Choose 'equal_size', 'equal_number_of_bins' or 'manual'")
            # Check category counts and warn if necessary, after binning
            check_category_counts_and_warn(data, column)
        else:
            # Check category counts and warn if necessary
            check_category_counts_and_warn(data, column)

        # Encoding values using the appropriate encoding_method
        group = data.groupby(column)[target_column]

        if encoding_method == 'mean':
            encoding = group.mean()
        elif encoding_method == 'sum':
            encoding = group.sum()
        elif encoding_method == 'median':
            encoding = group.median()
        elif encoding_method == 'quantile_25':
            encoding = group.quantile(0.25)
        elif encoding_method == 'quantile_75':
            encoding = group.quantile(0.75)
        elif encoding_method == 'count':
            encoding = group.count()
        elif encoding_method == 'smoothed_target_encoding':
            smoothing = 1
            encoding = (group.sum() + smoothing) / \
                       (group.count() + smoothing * 2)
        elif encoding_method == 'bayesian_target_encoding':
            prior_mean = data[target_column].mean()
            m = 5
            encoding = (group.sum() + prior_mean * m) / (group.count() + m)
        else:
            raise ValueError(
                "encoding_method must be one of 'mean', 'sum', 'median', 'quantile_25', 'quantile_75', 'count',"
                "'smoothed_target_encoding', or 'bayesian_target_encoding'")

        encoded_column_name = f'{column}_encoded'
        data[encoded_column_name] = data[column].map(encoding)

        # Removing original columns from the data
        data.drop(column, axis=1, inplace=True)

        # Renaming the encoded column to the original column name
        data.rename(columns={encoded_column_name: column}, inplace=True)

        # Saving statistics
        target_encoding_dict[column]['encoding_values'] = encoding.to_dict()

    return data, target_encoding_dict


# Apply Target Encoding using dictionary
def apply_target_encoding_using_dict(data: pd.DataFrame, config: Dict, target_encoding_dict: dict) -> pd.DataFrame:
    """
    Applies target encoding to the provided dataframe using the target encoding dictionary.

    Args:
        data: The input dataframe to be processed.
        config: A dictionary containing the rules for target encoding.
        target_encoding_dict: A dictionary containing the target encoding values from the training data.

    Raises:
        ValueError: If the specified column is not found in the data or if there are values in the column that were not
            found in the target encoding dictionary.
        ValueError: If any column configuration doesn't contain all required keys.

    Returns:
        A dataframe with target encoding applied.
    """
    target_encoding_config = config['target_encoding']

    if not target_encoding_config or not any(target_encoding_config):
        return data

    required_keys = ['column', 'type']

    for column_config in target_encoding_config:
        if not column_config:
            continue

        # Check if all required keys are in the configuration
        if not all(key in column_config for key in required_keys):
            raise ValueError(
                f'Each column configuration must contain the keys {required_keys}')

        column = column_config['column']
        column_type = column_config['type']

        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")

        # Initialize a new DataFrame to avoid SettingWithCopyWarning
        data = data.copy()

        if column_type == 'continuous':
            # Apply binning for continuous variables using saved bins
            bins = target_encoding_dict[column]['bins']
            min_value, max_value = data[column].min(), data[column].max()

            if min_value < min(bins) or max_value > max(bins):
                raise ValueError(
                    f"Values in column '{column}' exceed the defined bins")

            data[column] = pd.cut(data[column], bins=bins,
                                  labels=False, include_lowest=True)

        encoding = target_encoding_dict[column]['encoding_values']

        # Check if the value exists in the encoding dictionary, excluding NaN values
        unique_values = data[column].unique()
        missing_values = [
            value for value in unique_values if value not in encoding and not pd.isna(value)]

        if missing_values:
            raise ValueError(
                f"Values {missing_values} in column '{column}' were not found in data")

        # Applying Target Encoding to input data using training set statistics
        data[f'{column}_target_encoded'] = data[column].map(encoding)

        # Removing original columns from the data
        data.drop(column, axis=1, inplace=True)

        # Renaming the encoded column to the original column name
        data.rename(columns={f'{column}_target_encoded': column}, inplace=True)

    return data


# Calculate Information Value (IV)
def calculate_iv_and_warn(positive: pd.Series, negative: pd.Series, total_positives: int, total_negatives: int,
                          column: str):
    """
    Calculates the Information Value (IV) for a given column and provides warnings if the information is too low or
    too high.

    Args:
        positive: A series containing the count of positive target values for each unique value in the column.
        negative: A series containing the count of negative target values for each unique value in the column.
        total_positives: The total number of positive instances in the data.
        total_negatives: The total number of negative instances in the data.
        column: The name of the column.

    Returns:
        The Information Value (IV) for the column.
    """
    # Calculate the proportion of each group in positives and negatives
    positive_proportion = positive / total_positives
    negative_proportion = negative / total_negatives

    # Calculate the WOE values
    woe_values = np.log(positive_proportion / negative_proportion)

    # Calculate the information value (IV) for each unique value in the column
    iv_values = (positive_proportion - negative_proportion) * woe_values

    # Sum all IVs to get the overall IV for the column
    iv = round(iv_values.sum(), 3)

    if iv < 0.02:
        print(
            f"Warning: The variable '{column}' is not informative (IV = {iv})")
    elif 0.02 >= iv <= 0.1:
        print(
            f"Warning: The variable '{column}' is poorly informative (IV = {iv})")
    elif iv > 0.5:
        print(
            f"Warning: The variable '{column}' might be too informative (IV = {iv}), suggesting potential overfitting")

    return iv


# Apply WOE Encoding
def apply_woe_encoding(data: pd.DataFrame, config: Dict, full_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply WOE Encoding to the DataFrame.

    Args:
        data: The input DataFrame.
        config: Configuration dictionary containing rules for applying WOE encoding.
        full_data: The DataFrame before splitting to the training and testing sets to pass to get_min_max_values().

    Raises:
        ValueError: If the specified column is not found in the data, if the target column is not found in the data,
            or if an invalid binning method is provided.
        ValueError: If any column configuration doesn't contain all required keys.

    Returns:
        Tuple containing the transformed DataFrame and a dictionary with the WOE encoding statistics.
    """
    woe_encoding_config = config['woe_encoding']
    woe_encoding_dict = {}

    if not woe_encoding_config or not any(woe_encoding_config):
        return data, woe_encoding_dict

    required_keys = ['column', 'type', 'target_column']

    for column_config in woe_encoding_config:
        if not column_config:
            continue

        # Check if all required keys are in the configuration
        if not all(key in column_config for key in required_keys):
            raise ValueError(
                f'Each column configuration must contain the keys {required_keys}')

        column = column_config['column']
        column_type = column_config['type']
        target_column = column_config['target_column']

        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")

        if target_column not in data.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in data")

        woe_encoding_dict[column] = {}

        # Initialize a new DataFrame to avoid SettingWithCopyWarning
        data = data.copy()

        # Apply binning for continuous variables
        if column_type == 'continuous':
            binning_method = column_config.get(
                'binning_method', 'equal_number_of_bins')

            if binning_method in ('equal_size', 'manual'):
                validate_manual_bins(full_data, column_config)
                min_value, max_value = get_min_max_values(
                    full_data, column_config)
                data, bins = apply_binning(
                    data, column_config, min_value, max_value)
                woe_encoding_dict[column]['bins'] = bins

            elif binning_method == 'equal_number_of_bins':
                min_value, max_value = get_min_max_values(
                    full_data, column_config)
                data, bins = apply_binning(
                    data, column_config, min_value, max_value)
                woe_encoding_dict[column]['bins'] = bins

            else:
                raise ValueError(
                    "Invalid method. Choose 'equal_size', 'equal_number_of_bins' or 'manual'")

        # Calculate WOE values
        total_positives = (data[target_column] == 1).sum()
        total_negatives = (data[target_column] == 0).sum()

        group = data.groupby(column)[target_column]
        positive = group.apply(lambda x: (x == 1).sum() + 1)
        negative = group.apply(lambda x: (x == 0).sum() + 1)

        woe_values = (positive / total_positives) / \
                     (negative / total_negatives)
        woe_values = woe_values.apply(lambda x: np.log(x))

        encoded_column_name = f'{column}_woe_encoded'
        data[encoded_column_name] = data[column].map(woe_values)

        # Checking the Information Value (IV) for the column
        calculate_iv_and_warn(positive, negative,
                              total_positives, total_negatives, column)

        # Removing original columns from the data
        data.drop(column, axis=1, inplace=True)

        # Renaming the encoded column to the original column name
        data.rename(columns={encoded_column_name: column}, inplace=True)

        # Saving WOE values
        woe_encoding_dict[column]['woe_values'] = woe_values.to_dict()

    return data, woe_encoding_dict


# Apply WOE Encoding using dictionary
def apply_woe_encoding_using_dict(data: pd.DataFrame, config: Dict, woe_encoding_dict: Dict) -> pd.DataFrame:
    """
    Applies WOE encoding to the provided dataframe using the WOE encoding dictionary.

    Args:
        data: The input dataframe to be processed.
        config: A dictionary containing the rules for WOE encoding.
        woe_encoding_dict: A dictionary containing the WOE encoding values from the training data.

    Raises:
        ValueError: If the specified column is not found in the data or if a value in the column is not found
            in the encoding dictionary.
        ValueError: If any column configuration doesn't contain all required keys.

    Returns:
        A dataframe with WOE encoding applied.
    """
    woe_encoding_config = config['woe_encoding']

    if not woe_encoding_config or not any(woe_encoding_config):
        return data

    required_keys = ['column', 'type']

    for column_config in woe_encoding_config:
        if not column_config:
            continue

        # Check if all required keys are in the configuration
        if not all(key in column_config for key in required_keys):
            raise ValueError(
                f'Each column configuration must contain the keys {required_keys}')

        column = column_config['column']
        column_type = column_config['type']

        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")

        # Initialize a new DataFrame to avoid SettingWithCopyWarning
        data = data.copy()

        if column_type == 'continuous':
            # Apply binning for continuous variables using saved bins
            bins = woe_encoding_dict[column]['bins']
            data[column] = pd.cut(data[column], bins=bins,
                                  labels=False, include_lowest=True)

        encoding = woe_encoding_dict[column]['woe_values']

        # Check if the value exists in the encoding dictionary, excluding NaN values
        unique_values = data[column].unique()
        missing_values = [
            value for value in unique_values if value not in encoding and not pd.isna(value)]

        if missing_values:
            raise ValueError(
                f"Values {missing_values} in column '{column}' were not found in the training data")

        # Applying WOE Encoding to input data using training set statistics
        data[f'{column}_woe_encoded'] = data[column].map(encoding)

        # Removing original columns from the data
        data.drop(column, axis=1, inplace=True)

        # Renaming the encoded column to the original column name
        data.rename(columns={f'{column}_woe_encoded': column}, inplace=True)

    return data


# Calculate class ratio
def calculate_class_ratio(data: pd.DataFrame, config: Dict) -> float:
    """
    Check if the target classes in the DataFrame are balanced.

    Args:
        data: The input DataFrame.
        config: Configuration dictionary containing the target column name.

    Raises:
        ValueError: If the target column is not found in the data or if the target column is not binary.
        ValueError: If the configuration doesn't contain all required keys.

    Returns:
        The actual class ratio.
    """
    required_keys = ['target_column']

    # Check if all required keys are in the configuration
    if not all(key in config for key in required_keys):
        raise ValueError(
            f'Configuration must contain the keys {required_keys}')

    target_column = config['target_column']

    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")

    class_counts = data[target_column].value_counts()

    # Checking if the target column is binary
    if len(class_counts) != 2:
        raise ValueError('The target column is not binary')

    class_ratio = class_counts.max() / class_counts.min()

    return class_ratio


# Resample data
def resample_data(data: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Resample the DataFrame to balance the target classes.

    Args:
        data: The input DataFrame.
        config: Configuration dictionary containing the target column name.

    Raises:
        ValueError: If the target column is not found in the data or if an invalid resampling method is specified.
        ValueError: If the configuration doesn't contain all required keys.

    Returns:
        The resampled DataFrame.
    """
    required_keys = ['target_column', 'resampling_method', 'random_state']

    # Check if all required keys are in the configuration
    if not all(key in config for key in required_keys):
        raise ValueError(
            f'Configuration must contain the keys {required_keys}')

    target_column = config['target_column']
    method = config['resampling_method']
    random_state = config['random_state']

    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")

    # Initialize a new DataFrame to avoid SettingWithCopyWarning
    data = data.copy()

    x = data.drop(target_column, axis=1)
    y = data[target_column]

    if method == 'oversample':
        resampler = RandomOverSampler(random_state=random_state)
    elif method == 'undersample':
        resampler = RandomUnderSampler(random_state=random_state)
    else:
        raise ValueError(
            "Invalid method. Choose 'oversample' or 'undersample'")

    x_res, y_res = resampler.fit_resample(x, y)

    # Create a new dataframe from resampled data
    resampled_data = pd.DataFrame(x_res, columns=x.columns)
    resampled_data[target_column] = y_res

    # Reindex columns to match the original order
    resampled_data = resampled_data.reindex(columns=data.columns)

    return resampled_data


# Scale data
def scale_data(data: pd.DataFrame, config: Dict) -> \
        Tuple[pd.DataFrame, Optional[Union[StandardScaler, RobustScaler, MinMaxScaler]]]:
    """
    Scales the input dataframe using the specified scaler type.

    Args:
        data: The features dataframe.
        config: Configuration dictionary containing the type of scaler to use and the columns to be scaled.

    Raises:
        ValueError: If an invalid scaler_type is specified or if the specified column is not found in the data.
        ValueError: If the scaling configuration doesn't contain all required keys.

    Returns:
        The scaled dataframe and the scaler used.
    """
    scaling_config = config['scaling']

    required_keys = ['scaler_type', 'columns']

    # Check if all required keys are in the configuration
    if not all(key in scaling_config for key in required_keys):
        raise ValueError(
            f'Configuration must contain the keys {required_keys}')

    scaler_type = scaling_config['scaler_type']
    columns = scaling_config['columns']

    for column in columns:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")

    # Initialize a new DataFrame to avoid SettingWithCopyWarning
    data = data.copy()

    data_to_scale = data[columns]

    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'none':
        return data, None
    else:
        raise ValueError(
            f"Invalid scaler_type: '{scaler_type}'. Valid options are 'standard', 'robust', 'minmax', and 'none'.")

    data_scaled_values = scaler.fit_transform(data_to_scale)
    data_scaled = pd.DataFrame(
        data_scaled_values, columns=columns, index=data.index)

    # Update the original data frame with the scaled data
    data.update(data_scaled)

    return data, scaler


# Scale data using a scaler
def scale_data_using_scaler(data: pd.DataFrame, config: Dict,
                            scaler: Optional[Union[StandardScaler, RobustScaler, MinMaxScaler]]) -> pd.DataFrame:
    """
    Scales the input dataframe using a given scaler.

    Args:
        data: The input dataframe to be scaled.
        config: Configuration dictionary containing the columns to be scaled.
        scaler: The scaler object.

    Raises:
        ValueError: If the specified column is not found in the data.
        ValueError: If the scaling configuration doesn't contain all required keys.

    Returns:
        A dataframe with the data scaled using the given scaler.
    """
    if not scaler:
        return data

    scaling_config = config['scaling']

    required_keys = ['columns']

    # Check if all required keys are in the configuration
    if not all(key in scaling_config for key in required_keys):
        raise ValueError(
            f'Configuration must contain the keys {required_keys}')

    columns = scaling_config['columns']

    for column in columns:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")

    # Initialize a new DataFrame to avoid SettingWithCopyWarning
    data = data.copy()

    data_to_scale = data[columns]
    data_scaled_values = scaler.transform(data_to_scale)
    data_scaled = pd.DataFrame(
        data_scaled_values, columns=columns, index=data.index)

    # Update the original data frame with the scaled data
    data.update(data_scaled)

    return data


# Split dataset into train and test sets
def split_data_into_sets(data: pd.DataFrame, config: Dict) -> tuple:
    """
    Splits the input dataframe into training and testing sets.

    Args:
        data: The input dataframe to be processed.
        config: A dictionary containing configuration settings for the split.

    Returns:
        A tuple containing the training and testing dataframes.
    """
    required_keys = ['test_size', 'random_state']

    # Check if all required keys are in the configuration
    if not all(key in config for key in required_keys):
        raise ValueError(
            f'Configuration must contain the keys {required_keys}')

    test_size = config['test_size']
    random_state = config['random_state']

    # Splitting data into training and test sets
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state)

    return train_data, test_data


# Split data into features and labels
def split_data_into_feats_and_labels(train_data: pd.DataFrame, test_data: pd.DataFrame, config: Dict) -> tuple:
    """
    Splits the input dataframes into features and labels.

    Args:
        train_data: The training dataframe.
        test_data: The testing dataframe.
        config: Configuration dictionary containing the target column name.

    Returns:
        A tuple containing the training and testing features and labels.
    """
    required_keys = ['target_column']

    # Check if all required keys are in the configuration
    if not all(key in config for key in required_keys):
        raise ValueError(
            f'Configuration must contain the keys {required_keys}')

    target_column = config['target_column']

    y_train = train_data[target_column]
    x_train = train_data.drop([target_column], axis=1)

    y_test = test_data[target_column]
    x_test = test_data.drop([target_column], axis=1)

    return x_train, y_train, x_test, y_test


# Compute font scale
def compute_font_scale(num_features: int, base_num_features: int = 20, base_font_scale: float = 0.7) -> float:
    """
    Computes the font scale for a heatmap plot in relation to the number of features.

    Args:
        num_features: The number of features.
        base_num_features: The number of features for which the base_font_scale was suitable. Default is 20.
        base_font_scale: The font scale that was suitable for the base_num_features. Default is 0.7.

    Returns:
        font_scale: The computed font scale.
    """
    font_scale = base_font_scale * (base_num_features / num_features)

    return font_scale


# Plot correlation matrix
def plot_correlation_matrix(train_data: pd.DataFrame, config: Dict, timestamp: str, path: str = '') -> None:
    """
    Plots a correlation matrix for the features in the training dataset and saves the plot as an image file.

    Args:
        train_data: The dataframe containing the training data.
        config: Configuration dictionary containing the tag and target column name.
        timestamp: The timestamp to be used in the generated file name.
        path: The path where the image file will be saved (optional).

    Raises:
        ValueError: If x_train contains NaN or infinite values, or if any variable in x_train is constant.

    Returns:
        None
    """
    required_keys = ['tag', 'target_column']

    # Check if all required keys are in the configuration
    if not all(key in config for key in required_keys):
        raise ValueError(
            f'Configuration must contain the keys {required_keys}')

    tag = config['tag']
    target_column = config['target_column']

    x_train = train_data.drop(columns=[target_column])
    numerical_x_train = x_train.select_dtypes(include=[np.number])

    # Check for NaN or infinite values in numerical_x_train
    if np.isnan(numerical_x_train.values).any() or np.isinf(numerical_x_train.values).any():
        raise ValueError('The input data contains NaN or infinite values, which will cause errors in computing '
                         'the correlation matrix.')

    # Check if any variables are constant
    unique_counts = numerical_x_train.apply(
        lambda col: len(col.unique()), axis=0)
    if (unique_counts == 1).any():
        constant_variables = unique_counts[unique_counts == 1].index.tolist()
        raise ValueError(f'One or more variables are constant (i.e., have only one unique value), which will cause '
                         f'errors in computing the correlation matrix. The constant variables are: '
                         f'{constant_variables}')

    # Compute correlation matrix
    corr = numerical_x_train.corr()
    feature_names = numerical_x_train.columns.tolist()

    # Create heatmap plot of the correlation matrix
    num_features = len(numerical_x_train.columns)
    font_scale = compute_font_scale(num_features)

    plt.figure(figsize=(10, 8))
    sns.set(font_scale=font_scale)
    sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f',
                xticklabels=feature_names, yticklabels=feature_names,
                linewidths=0.5, linecolor='gray')
    plt.title('Correlation Matrix for Train Data')

    # Save the plot as an image
    save_format = 'png'
    name_with_tag = f'train_data_features_correlation_matrix_{tag}'
    file_name = generate_file_name_with_timestamp(
        name_with_tag, save_format, timestamp, path)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()


# Generate filename with date and time
def generate_file_name_with_timestamp(file_name: str, file_ext: str, timestamp: str, path: str = '') -> str:
    """
    Generates a file name with an optional timestamp and path.

    Args:
        file_name: The base file name.
        file_ext: The file extension (without leading dot).
        timestamp: A string containing the timestamp to include in the file name, or an empty string to not include
                   a timestamp.
        path: The directory path where the file is to be saved (optional).

    Returns:
        A string containing the full path to the generated file name with the optional timestamp.
    """
    if not timestamp:
        file_name_with_timestamp = f'{file_name}.{file_ext}'
    else:
        file_name_with_timestamp = f'{file_name}_{timestamp}.{file_ext}'

    return os.path.join(path, file_name_with_timestamp)


# Save data to file
def save_object_to_file(obj: Any, file: str) -> None:
    """
    Saves an object to a file with the specified type.

    Args:
        obj: The object to save. The type must be compatible with the file extension.
        file: The name of the file where the object will be saved (extension determines the file type).

    Raises:
        ValueError: If an unsupported file extension is specified.
        TypeError: If the object type is incompatible with the file extension. Incompatibilities include:
                   - JSON: Object must be of type 'dict' or 'list'.
                   - H5: Object must be a Keras model.
                   - CSV: Object must be of type 'pd.DataFrame'.

    Returns:
        None
    """
    # Get file extension to determine file type
    _, file_extension = os.path.splitext(file)

    if file_extension == '.json':
        if not isinstance(obj, (dict, list)):
            raise TypeError(
                f"Object must be of type 'dict' or 'list' for JSON serialization. Got: {type(obj)}")
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=4)
    elif file_extension == '.pkl':
        with open(file, 'wb') as f:
            pickle.dump(obj, f)
    elif file_extension == '.h5':
        if not hasattr(obj, 'save_weights'):
            raise TypeError(
                f"Object must be a Keras model to save as '.h5'. Got: {type(obj)}")
        obj.save_weights(file)
    elif file_extension == '.csv':
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(
                f"Object must be of type 'pd.DataFrame' for CSV serialization. Got: {type(obj)}")
        obj.to_csv(file, index=False, encoding='UTF-8')
    else:
        raise ValueError(
            f"Invalid file extension. Supported extensions are '.json', '.pkl', '.h5', and '.csv'. Got: "
            f"{file_extension}")


# Save data and dictionaries to files
def save_data_and_dicts(train_data: pd.DataFrame, test_data: pd.DataFrame, config: Dict, timestamp: str,
                        num_stats_dict: Optional[Dict] = None, text_stats_dict: Optional[Dict] = None,
                        thresholds_dict: Optional[Dict] = None, replacement_values_dict: Optional[Dict] = None,
                        ohe_columns_dict: Optional[Dict] = None, transformation_params_dict: Optional[Dict] = None,
                        target_encoding_dict: Optional[Dict] = None, woe_encoding_dict: Optional[Dict] = None,
                        scaler: Optional[Union[StandardScaler,
                                               RobustScaler, MinMaxScaler]] = None,
                        save_as_json: Optional[bool] = False, path: str = '') -> None:
    """
    Save train and test data and various dictionaries to files.

    Args:
        train_data (pd.DataFrame): The training dataset.
        test_data (pd.DataFrame): The test dataset.
        config (Dict): The configuration dictionary containing the 'tag' key.
        timestamp (str): The timestamp to include in the file names.
        num_stats_dict (Optional[Dict], default=None): The numerical statistics dictionary.
        text_stats_dict (Optional[Dict], default=None): The text statistics dictionary.
        thresholds_dict (Optional[Dict], default=None): The thresholds dictionary.
        replacement_values_dict (Optional[Dict], default=None): The replacement values dictionary.
        ohe_columns_dict (Optional[Dict], default=None): The one-hot encoding columns dictionary.
        transformation_params_dict (Optional[Dict], default=None): The transformation parameters dictionary.
        target_encoding_dict (Optional[Dict], default=None): The target encoding dictionary.
        woe_encoding_dict (Optional[Dict], default=None): The Weight of Evidence (WOE) encoding dictionary.
        scaler (Optional[Union[StandardScaler, RobustScaler, MinMaxScaler]], default=None): The data scaler.
        save_as_json (Optional[bool], default=False): Whether to also save the data in json format.
        path (str, default=''): The path where the files will be saved.

    Raises:
        ValueError: If 'tag' is not in the configuration, or if train_data and test_data are None.

    Returns:
        None
    """
    # Check if all required keys are in the configuration
    if 'tag' not in config:
        raise ValueError('Configuration must contain the key "tag"')

    tag = config['tag']

    # Check if train_data and test_data are not None
    if train_data is None or test_data is None:
        raise ValueError('train_data and test_data cannot be None')

    # Create dictionary with all the input data
    data_dict = {
        'num_stats_dict': num_stats_dict,
        'text_stats_dict': text_stats_dict,
        'thresholds_dict': thresholds_dict,
        'replacement_values_dict': replacement_values_dict,
        'ohe_columns_dict': ohe_columns_dict,
        'transformation_params_dict': transformation_params_dict,
        'target_encoding_dict': target_encoding_dict,
        'woe_encoding_dict': woe_encoding_dict,
        'scaler': scaler
    }

    # Configuration of formats to save data
    save_config = {
        'csv': ['train_data', 'test_data'],
        'pkl': ['num_stats_dict', 'text_stats_dict', 'thresholds_dict', 'replacement_values_dict',
                'ohe_columns_dict', 'transformation_params_dict', 'target_encoding_dict',
                'woe_encoding_dict', 'scaler'],
        'json': ['num_stats_dict', 'text_stats_dict', 'thresholds_dict', 'replacement_values_dict',
                 'ohe_columns_dict', 'transformation_params_dict', 'target_encoding_dict',
                 'woe_encoding_dict']
    }

    # Save train_data and test_data as they should always have content
    for save_format in ['csv']:
        for name in ['train_data', 'test_data']:
            # Get data
            data = locals()[name]

            # Generate name with tag
            name_with_tag = f'{name}_{tag}'

            # Save data to file
            file_name = generate_file_name_with_timestamp(
                name_with_tag, save_format, timestamp, path)
            save_object_to_file(data, file_name)

    # Iterate over save formats
    for save_format, save_names in save_config.items():
        for name in save_names:
            # Get data
            data = data_dict.get(name)

            # If data is not None or an empty dictionary
            if data and data != {}:
                # Generate name with tag
                name_with_tag = f'{name}_{tag}'

                # Save data to file
                file_name = generate_file_name_with_timestamp(
                    name_with_tag, save_format, timestamp, path)
                save_object_to_file(data, file_name)

                # Save JSON version if required
                if save_as_json and save_format == 'pkl' and name in save_config['json']:
                    file_name_json = generate_file_name_with_timestamp(
                        name_with_tag, 'json', timestamp, path)
                    save_object_to_file(data, file_name_json)


# Load data from file
def load_data_from_file(file: str, keras_weights: bool = False,
                        ignore_error: bool = False) -> Any:
    """
    Loads data from a file with the specified type.

    Args:
        file: The name of the file to load data from.
        keras_weights: Whether the file contains Keras weights (only applicable to JSON files).
        ignore_error: Whether to ignore errors and return None instead of raising an exception.

    Raises:
        ValueError: If an invalid file_type is specified.
        FileNotFoundError: If the file is not found and ignore_error is False.

    Returns:
        The loaded data, or None if an error occurs and ignore_error is True.
    """
    # Get file extension to determine file type
    _, file_extension = os.path.splitext(file)

    try:
        if file_extension == '.json':
            with open(file, 'r') as file:
                if keras_weights:
                    model_json = file.read()
                    data = model_from_json(model_json)
                else:
                    data = json.load(file)
        elif file_extension == '.pkl':
            with open(file, 'rb') as file:
                data = pickle.load(file)
        elif file_extension == '.h5':
            if keras_weights:
                raise ValueError(
                    "Please use 'load_weights()' method to load Keras weights from an h5 file.")
            else:
                raise ValueError(
                    'Loading data from h5 files is not supported in this function.')
        else:
            raise ValueError(
                f"Invalid file extension. Supported extensions are '.json', '.pkl', and '.h5'."
                f" Got: {file_extension}")

        return data
    except FileNotFoundError as e:
        if ignore_error:
            warnings.warn(f"File '{file}' not found. Returning None.")
            return None
        else:
            raise e from None
    except Exception as e:
        if ignore_error:
            warnings.warn(
                f"Error occurred while loading file '{file}'. Returning None. Error message: {str(e)}")
            return None
        else:
            raise e from None


# Check missing and non-numerical values
def check_missing_and_non_numerical_values(data: pd.DataFrame) -> None:
    """
    Check the DataFrame for missing values and non-numerical values.

    Args:
        data: The input DataFrame.

    Returns:
        None. Warnings are issued if there are any missing values or non-numerical values in the DataFrame.
    """
    missing_values_columns = []
    non_numerical_values_columns = []

    for column in data.columns:
        if data[column].isnull().any():
            missing_values_columns.append(column)
        if not pd.api.types.is_numeric_dtype(data[column]):
            non_numerical_values_columns.append(column)

    warning_message = ''
    if missing_values_columns:
        warning_message += f"The following columns have missing values: {', '.join(missing_values_columns)}. If you " \
            f"have applied binning, you can change the method to 'equal_number_of_bins' or 'manual'. " \
            f"This will allow you to create bins that include values from the test dataset without data" \
            f" leakage.\n"
    if non_numerical_values_columns:
        warning_message += f"The following columns have non-numerical values:" \
                           f" {', '.join(non_numerical_values_columns)}.\n"

    if warning_message:
        warnings.warn(warning_message)


# Validate columns match and order
def validate_columns_and_order(x_train: pd.DataFrame, x_test: pd.DataFrame) -> None:
    """
    Checks that the training and testing dataframes have the same columns in the same order.

    Args:
        x_train: The training features dataframe.
        x_test: The testing features dataframe.

    Raises:
        ValueError: If the dataframes do not have the same columns.
        ValueError: If the dataframes do not have the columns in the same order.
    """
    train_columns_set = set(x_train.columns)
    test_columns_set = set(x_test.columns)

    if train_columns_set != test_columns_set:
        missing_in_test = train_columns_set - test_columns_set
        missing_in_train = test_columns_set - train_columns_set

        error_message = ''

        if missing_in_test:
            error_message += f"The following columns are present in the training data but not the testing data:" \
                             f" {', '.join(missing_in_test)}."

        if missing_in_train:
            error_message += f"The following columns are present in the testing data but not the training data:" \
                             f" {', '.join(missing_in_train)}."

        raise ValueError(error_message)

    # Check if columns are in the same order
    if list(x_train.columns) != list(x_test.columns):
        error_message = 'The columns in the training and testing data are not in the same order.'
        raise ValueError(error_message)


# Apply data preprocessing
def apply_data_preprocessing(train_data: pd.DataFrame, config: Dict, full_data: pd.DataFrame) \
        -> Tuple[pd.DataFrame, Dict, Dict, Dict, Dict, Dict, Dict, Dict, Dict,
                 Optional[Union[StandardScaler, RobustScaler, MinMaxScaler]]]:
    """
    Apply preprocessing steps to data according to a config dictionary. Returns preprocessed datasets along with
     dictionaries of various preprocessing parameters and values, the scaler if data scaling was applied, and the
     column order.

    Args:
        train_data (pd.DataFrame): The training dataset.
        config (Dict): The configuration dictionary containing steps and associated actions.
        full_data: The DataFrame before splitting to the training and testing sets.

    Returns:
        Tuple[pd.DataFrame, Dict, Dict, Dict, Dict, Dict, Dict, Dict, Dict,
         Optional[Union[StandardScaler, RobustScaler, MinMaxScaler]]]:
        The preprocessed training dataset, dictionaries of various preprocessing parameters and values,
        if they were created during the process, the scaler if data scaling was applied, and the column order.

    Raises:
        ValueError: If any of the steps is not recognized.
    """
    num_stats_dict, text_stats_dict, thresholds_dict, replacement_values_dict = {}, {}, {}, {}
    ohe_columns_dict, transformation_params_dict, target_encoding_dict, woe_encoding_dict = {}, {}, {}, {}
    scaler = None

    steps = config['preprocessing']['steps']
    actions = config['preprocessing']['actions']

    # Dropping columns in the training set
    train_data = drop_columns(train_data, config)

    # Processing the training set step by step
    for step in steps:
        if step not in actions:
            raise ValueError(f"Step '{step}' not found in actions.")

        action_config = actions[step]
        action_config_key = list(action_config.keys())[0]

        # Filling in missing numeric values with zeroes in the training set
        if action_config_key == 'filling_missing_numeric_values_with_zeroes':
            # We use full_data in subsequent operations and for these operations the contents of full_data
            # must be processed in the same way as train_data
            full_data = fill_missing_numeric_values_with_zeros(
                full_data, action_config)
            train_data = fill_missing_numeric_values_with_zeros(
                train_data, action_config)

        # Filling in missing numeric values using specified statistical methods in the training sets
        elif action_config_key == 'filling_missing_numeric_values_with_stats_method':
            # We use full_data in subsequent operations and for these operations the contents of full_data
            # must be processed in the same way as train_data
            full_data, _ = fill_missing_numeric_values_with_stats_method(full_data, action_config,
                                                                         action_config_key)
            train_data, num_stats_dict = fill_missing_numeric_values_with_stats_method(train_data, action_config,
                                                                                       action_config_key)

        # Filling in missing text values using specified methods in the training set
        elif action_config_key == 'filling_missing_text_values':
            # We use full_data in subsequent operations and for these operations the contents of full_data
            # must be processed in the same way as train_data
            full_data, _ = fill_missing_text_values_with_method(full_data, action_config,
                                                                action_config_key)
            train_data, text_stats_dict = fill_missing_text_values_with_method(train_data, action_config,
                                                                               action_config_key)

        # Removing outliers in the training set
        elif action_config_key == 'removing_outliers':
            # We use full_data in subsequent operations and for these operations the contents of full_data
            # must be processed in the same way as train_data
            full_data, _ = remove_outliers(
                full_data, action_config)
            train_data, thresholds_dict = remove_outliers(
                train_data, action_config)

        # Replacing outliers in the training set
        elif action_config_key == 'replacing_outliers':
            # We use full_data in subsequent operations and for these operations the contents of full_data
            # must be processed in the same way as train_data
            full_data, _ = replace_outliers(
                full_data, action_config)
            train_data, replacement_values_dict = replace_outliers(
                train_data, action_config)

        # Converting data to numeric values in the training set
        elif action_config_key == 'converting_to_number':
            # We use full_data in subsequent operations and for these operations the contents of full_data
            # must be processed in the same way as train_data
            full_data = convert_to_number(
                full_data, action_config)
            train_data = convert_to_number(
                train_data, action_config)

        # Applying One-Hot Encoding to the training set
        elif action_config_key == 'one_hot_encoding':
            train_data, ohe_columns_dict = apply_one_hot_encoding(
                train_data, action_config, full_data)

        # Applying Ordinal Encoding to the training set
        elif action_config_key == 'ordinal_encoding':
            train_data = apply_ordinal_encoding(train_data, action_config)

        # Transforming data in the training set
        elif action_config_key == 'transformation':
            # We use full_data in subsequent operations and for these operations the contents of full_data
            # must be processed in the same way as train_data
            full_data, _ = transform_data(
                full_data, action_config)
            train_data, transformation_params_dict = transform_data(
                train_data, action_config)

        # Applying Target Encoding to the training set
        elif action_config_key == 'target_encoding':
            train_data, target_encoding_dict = apply_target_encoding(
                train_data, action_config, full_data)

        # Applying WOE Encoding to the training set
        elif action_config_key == 'woe_encoding':
            train_data, woe_encoding_dict = apply_woe_encoding(
                train_data, action_config, full_data)

        # Resampling data in the training set
        elif action_config_key == 'resampling':
            train_data = resample_data(
                train_data, action_config[action_config_key])

        # Scaling data in the training set
        elif action_config_key == 'scaling':
            train_data, scaler = scale_data(train_data, action_config)

        else:
            raise ValueError(
                f"Preprocessing step '{action_config_key}' not recognized.")

    # Ordering columns in the training set
    train_data = reorder_columns(train_data, config, ohe_columns_dict)

    return train_data, num_stats_dict, text_stats_dict, thresholds_dict, replacement_values_dict, ohe_columns_dict, \
        transformation_params_dict, target_encoding_dict, woe_encoding_dict, scaler


# Apply data preprocessing using dictionaries
def apply_data_preprocessing_using_dicts(test_data: pd.DataFrame, config: Dict, num_stats_dict: Dict,
                                         text_stats_dict: Dict, thresholds_dict: Dict, replacement_values_dict: Dict,
                                         ohe_columns_dict: Dict, transformation_params_dict: Dict,
                                         target_encoding_dict: Dict, woe_encoding_dict: Dict,
                                         scaler: Optional[Union[StandardScaler, RobustScaler, MinMaxScaler]]) -> \
        pd.DataFrame:
    """
    Applies preprocessing steps to the testing dataset based on a configuration dictionary and various preprocessing
    dictionaries. Returns the preprocessed dataset.

    Args:
        test_data (pd.DataFrame): The testing dataset.
        config (Dict): Configuration dictionary containing steps and associated actions.
        num_stats_dict (Dict): Dictionary with statistics for numeric values.
        text_stats_dict (Dict): Dictionary with statistics for text values.
        thresholds_dict (Dict): Dictionary with thresholds for preprocessing steps.
        replacement_values_dict (Dict): Dictionary with replacement values for preprocessing steps.
        ohe_columns_dict (Dict): Dictionary with one-hot encoding columns.
        transformation_params_dict (Dict): Dictionary with parameters for data transformation steps.
        target_encoding_dict (Dict): Dictionary with target encoding information.
        woe_encoding_dict (Dict): Dictionary with Weight of Evidence (WOE) encoding information.
        scaler (Optional[Union[StandardScaler, RobustScaler, MinMaxScaler]]): Scaler object if scaling was applied.

    Returns:
        pd.DataFrame: The preprocessed testing dataset.

    Raises:
        ValueError: If any of the steps is not recognized.
    """
    steps = config['preprocessing']['steps']
    actions = config['preprocessing']['actions']

    # Drop columns in the testing set
    test_data = drop_columns(test_data, config)

    # Process the testing set step by step
    for step in steps:
        if step not in actions:
            raise ValueError(f"Step '{step}' not found in actions.")

        action_config = actions[step]
        action_config_key = list(action_config.keys())[0]

        # Filling in missing numeric values with zeroes in the testing set
        if action_config_key == 'filling_missing_numeric_values_with_zeroes':
            test_data = fill_missing_numeric_values_with_zeros(
                test_data, action_config)

        # Filling in missing numeric values using specified statistical methods in the testing set
        elif action_config_key == 'filling_missing_numeric_values_with_stats_method':
            test_data = fill_missing_numeric_values_with_stats_method_using_dict(test_data, action_config,
                                                                                 action_config_key, num_stats_dict)

        # Filling in missing text values using specified methods in the testing set
        elif action_config_key == 'filling_missing_text_values':
            test_data = fill_missing_text_values_with_method_using_dict(test_data, action_config, action_config_key,
                                                                        text_stats_dict)

        # Removing outliers in the testing set
        elif action_config_key == 'removing_outliers':
            test_data = remove_outliers_using_dict(
                test_data, action_config, thresholds_dict)

        # Replacing outliers in the testing set
        elif action_config_key == 'replacing_outliers':
            test_data = replace_outliers_using_dict(
                test_data, action_config, replacement_values_dict)

        # Converting data to numeric values in the testing set
        elif action_config_key == 'converting_to_number':
            test_data = convert_to_number(
                test_data, action_config)

        # Applying One-Hot Encoding to the testing set
        elif action_config_key == 'one_hot_encoding':
            test_data = apply_one_hot_encoding_using_dict(
                test_data, action_config, ohe_columns_dict)

        # Applying Ordinal Encoding to the testing set
        elif action_config_key == 'ordinal_encoding':
            test_data = apply_ordinal_encoding(test_data, action_config)

        # Transforming data in the testing set
        elif action_config_key == 'transformation':
            test_data = transform_data_using_dict(
                test_data, action_config, transformation_params_dict)

        # Applying Target Encoding to the testing set
        elif action_config_key == 'target_encoding':
            test_data = apply_target_encoding_using_dict(
                test_data, action_config, target_encoding_dict)

        # Applying WOE Encoding to the testing set
        elif action_config_key == 'woe_encoding':
            test_data = apply_woe_encoding_using_dict(
                test_data, action_config, woe_encoding_dict)

        # Skip resampling step for the testing set
        elif action_config_key == 'resampling':
            pass

        # Scaling data in the testing set
        elif action_config_key == 'scaling':
            test_data = scale_data_using_scaler(
                test_data, action_config, scaler)

        else:
            raise ValueError(
                f"Preprocessing step '{action_config_key}' not recognized.")

    # Reorder columns in the testing set
    test_data = reorder_columns(test_data, config, ohe_columns_dict)

    return test_data


if __name__ == '__main__':
    # Parsing command-line arguments
    parser = argparse.ArgumentParser(description='Preprocess data.')
    parser.add_argument('data_file_path', help='Path to the CSV dataset file')
    parser.add_argument('-d', '--delimiter', default=',',
                        help="Delimiter used in the CSV file (default: ',')")
    parser.add_argument('config_file_path',
                        help='Path to the JSON configuration file')

    args = parser.parse_args()

    # Loading data from CSV file
    data = pd.read_csv(args.data_file_path,
                       delimiter=args.delimiter, encoding='UTF-8')

    # Loading config from JSON file
    ML_CONFIG = load_data_from_file(args.config_file_path)

    # Splitting data into training and test sets
    train_data, test_data = split_data_into_sets(
        data, ML_CONFIG)

    # Applying data preprocessing steps
    train_data, num_stats_dict, text_stats_dict, thresholds_dict, replacement_values_dict, ohe_columns_dict, \
        transformation_params_dict, target_encoding_dict, woe_encoding_dict, scaler = apply_data_preprocessing(
            train_data, ML_CONFIG, data)

    test_data = apply_data_preprocessing_using_dicts(test_data, ML_CONFIG, num_stats_dict, text_stats_dict,
                                                     thresholds_dict, replacement_values_dict, ohe_columns_dict,
                                                     transformation_params_dict, target_encoding_dict,
                                                     woe_encoding_dict, scaler)

    # Checking the result
    check_missing_and_non_numerical_values(train_data)
    check_missing_and_non_numerical_values(test_data)
    validate_columns_and_order(train_data, test_data)

    # Getting the directory of the input file
    data_file_dir = os.path.dirname(args.data_file_path)

    # Set current time
    # current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_time = ''

    # Plotting correlation matrix
    plot_correlation_matrix(train_data, ML_CONFIG,
                            current_time, path=data_file_dir)

    # Saving data to files
    save_data_and_dicts(train_data, test_data, ML_CONFIG, current_time, num_stats_dict, text_stats_dict,
                        thresholds_dict, replacement_values_dict, ohe_columns_dict, transformation_params_dict,
                        target_encoding_dict, woe_encoding_dict, scaler, save_as_json=True, path=data_file_dir)
