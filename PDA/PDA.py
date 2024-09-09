import pandas as pd
import logging
from pandas.api.types import is_numeric_dtype
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

import os
import shutil
from pathlib import Path


# Set up logging configuration
logging.basicConfig(
    filename='data_processing.log',  # Log file location
    level=logging.INFO,              # Logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S'      # Date format in logs
)

# Function to read a file into a DataFrame
def read_file(file_path, file_type='csv'):
    """
    Read a file into a DataFrame and return the DataFrame.
    
    Supported file types:
    - csv: Comma-separated values
    - xlsx: Excel files
    - json: JSON formatted data
    - parquet: Parquet files
    - hdf: HDF5 format
    - feather: Feather binary format

    Parameters:
    - file_path: str, the path to the file.
    - file_type: str, the file type to read ('csv', 'xlsx', 'json', 'parquet', 'hdf', 'feather').

    Returns:
    - DataFrame containing the data from the file.
    """
    logging.info(f"Reading file {file_path} of type {file_type}")
    try:
        if file_type.lower() == 'csv':
            df = pd.read_csv(file_path)
            logging.info(f"Successfully loaded CSV file: {file_path}")
        elif file_type.lower() == 'xlsx':
            df = pd.read_excel(file_path)
            logging.info(f"Successfully loaded Excel file: {file_path}")
        elif file_type.lower() == 'json':
            df = pd.read_json(file_path)
            logging.info(f"Successfully loaded JSON file: {file_path}")
        elif file_type.lower() == 'parquet':
            df = pd.read_parquet(file_path)
            logging.info(f"Successfully loaded Parquet file: {file_path}")
        elif file_type.lower() == 'hdf':
            df = pd.read_hdf(file_path)
            logging.info(f"Successfully loaded HDF file: {file_path}")
        elif file_type.lower() == 'feather':
            df = pd.read_feather(file_path)
            logging.info(f"Successfully loaded Feather file: {file_path}")
        else:
            raise ValueError("Unsupported file type. Use 'csv', 'xlsx', 'json', 'parquet', 'hdf', or 'feather'.")
        return df
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        raise


# Function to check for duplicates in a DataFrame
def check_duplicates(df, exact_match=True, match_columns=None, delete_duplicates=False):
    """
    Check a DataFrame for duplicate entries.
    
    Parameters:
    - df: DataFrame, the dataset.
    - exact_match: bool, whether to look for exact duplicates (default is True).
    - match_columns: list, specific columns to check for partial duplicates (used when exact_match is False).
    - delete_duplicates: bool, whether to delete duplicates (default is False).
    
    Returns:
    - DataFrame of duplicates or DataFrame without duplicates if delete_duplicates is True.
    """
    logging.info(f"Starting duplicate check (exact_match: {exact_match})")
    try:
        if exact_match:
            duplicates = df[df.duplicated(keep=False)]
            logging.info(f"Exact match duplicate check completed. Found {len(duplicates)} duplicates.")
        else:
            if not match_columns:
                raise ValueError("For partial matching, match_columns must be provided.")
            duplicates = df[df.duplicated(subset=match_columns, keep=False)]
            logging.info(f"Partial match duplicate check completed. Columns: {match_columns}. Found {len(duplicates)} duplicates.")
        
        if delete_duplicates:
            subset = match_columns if not exact_match else None
            df_cleaned = df.drop_duplicates(subset=subset)
            logging.info(f"Deleted {len(df) - len(df_cleaned)} duplicate rows.")
            return df_cleaned
        return duplicates
    except Exception as e:
        logging.error(f"Error during duplicate check: {e}")
        raise

# Function to check and handle null values
def check_nulls(df, fill_null=False, numeric_fill_value=0, text_fill_value='null'):
    """
    Check a DataFrame for null values and optionally fill them.
    
    Parameters:
    - df: DataFrame, the dataset.
    - fill_null: bool, whether to fill null values (default is False).
    - numeric_fill_value: numeric, the value to fill in numeric columns (default is 0).
    - text_fill_value: str, the value to fill in text columns (default is 'null').
    
    Returns:
    - DataFrame with filled null values if fill_null is True, otherwise returns a summary of null values.
    """
    logging.info(f"Starting null value check")
    if fill_null:
        try:
            for column in df.columns:
                if df[column].isnull().any():
                    if is_numeric_dtype(df[column]):
                        df[column].fillna(numeric_fill_value, inplace=True)
                        logging.info(f"Filled nulls in numeric column '{column}' with {numeric_fill_value}")
                    else:
                        df[column].fillna(text_fill_value, inplace=True)
                        logging.info(f"Filled nulls in text column '{column}' with '{text_fill_value}'")
            return df
        except Exception as e:
            logging.error(f"Error during null value filling: {e}")
            raise
    else:
        try:
            null_summary = df.isnull().sum()
            null_summary = null_summary[null_summary > 0]
            logging.info(f"Null value check completed. Found nulls in columns: {null_summary.index.tolist()}")
            return null_summary
        except Exception as e:
            logging.error(f"Error during null value check: {e}")
            raise

# Function to validate and convert data types
def validate_and_convert_dtypes(df, dtype_dict=None):
    """
    Validate and convert data types of columns in the DataFrame.
    
    Parameters:
    - df: DataFrame, the dataset.
    - dtype_dict: dict, mapping of column names to desired data types (default is None).
    
    Returns:
    - DataFrame with converted data types.
    """
    if dtype_dict:
        try:
            df = df.astype(dtype_dict)
            logging.info(f"Converted data types: {dtype_dict}")
        except Exception as e:
            logging.error(f"Error converting data types: {e}")
            raise
    return df

# Function to detect outliers
def detect_outliers(df, column, method='zscore', threshold=3):
    """
    Detect outliers in a numeric column using Z-score or IQR method.
    
    Parameters:
    - df: DataFrame, the dataset.
    - column: str, the column to check for outliers.
    - method: str, method of detection ('zscore' or 'iqr', default is 'zscore').
    - threshold: numeric, the threshold for outlier detection (default is 3 for Z-score, or a multiplier for IQR).
    
    Returns:
    - Series of boolean values indicating outliers.
    """
    try:
        if method == 'zscore':
            z_scores = stats.zscore(df[column].dropna())
            outliers = abs(z_scores) > threshold
        elif method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = (df[column] < (Q1 - threshold * IQR)) | (df[column] > (Q3 + threshold * IQR))
        else:
            raise ValueError("Unsupported method. Use 'zscore' or 'iqr'.")
        logging.info(f"Outlier detection completed for {column} using {method}. Found {outliers.sum()} outliers.")
        return outliers
    except Exception as e:
        logging.error(f"Error during outlier detection: {e}")
        raise

# Function to generate correlation matrix
def generate_correlation_matrix(df, method='pearson', plot=False):
    """
    Generate a correlation matrix for the dataset.
    
    Parameters:
    - df: DataFrame, the dataset.
    - method: str, correlation method ('pearson', 'kendall', or 'spearman', default is 'pearson').
    - plot: bool, whether to display a heatmap plot (default is False).
    
    Returns:
    - Correlation matrix as a DataFrame.
    """
    try:
        correlation_matrix = df.corr(method=method)
        logging.info(f"Generated {method} correlation matrix.")
        if plot:
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
            plt.title(f"{method.capitalize()} Correlation Matrix")
            plt.show()
        return correlation_matrix
    except Exception as e:
        logging.error(f"Error generating correlation matrix: {e}")
        raise

# Function to normalize/standardize data
def normalize_data(df, columns, method='minmax'):
    """
    Normalize or standardize the specified columns in the dataset.
    
    Parameters:
    - df: DataFrame, the dataset.
    - columns: list, the columns to normalize.
    - method: str, normalization method ('minmax' or 'zscore', default is 'minmax').
    
    Returns:
    - DataFrame with normalized/standardized columns.
    """
    try:
        if method == 'minmax':
            scaler = MinMaxScaler()
            df[columns] = scaler.fit_transform(df[columns])
            logging.info(f"Normalized columns {columns} using Min-Max scaling.")
        elif method == 'zscore':
            scaler = StandardScaler()
            df[columns] = scaler.fit_transform(df[columns])
            logging.info(f"Standardized columns {columns} using Z-score scaling.")
        else:
            raise ValueError("Unsupported method. Use 'minmax' or 'zscore'.")
        return df
    except Exception as e:
        logging.error(f"Error during normalization: {e}")
        raise

def convert_data_types(df, dtype_dict):
    """
    Convert columns in the DataFrame to specified data types.
    
    Parameters:
    - df: DataFrame, the dataset.
    - dtype_dict: dict, mapping of column names to desired data types.
    
    Returns:
    - DataFrame with converted data types.
    """
    try:
        df = df.astype(dtype_dict)
        logging.info(f"Data types converted as per {dtype_dict}.")
        return df
    except Exception as e:
        logging.error(f"Error during data type conversion: {e}")
        raise


def convert_and_copy_files(src_directory, dest_directory, target_file_type):
    """
    Convert or copy files from the source directory to the destination directory with a uniform file type.
    
    Parameters:
    - src_directory: str, the path to the source directory containing files to process.
    - dest_directory: str, the path to the destination directory where converted files will be saved.
    - target_file_type: str, the desired file type for all files in the destination directory ('csv', 'xlsx', 'json', 'parquet', 'hdf', 'feather').

    Returns:
    - None
    """
    logging.info(f"Starting file conversion in directory '{src_directory}' to '{target_file_type}' format.")
    
    # Create the destination directory if it doesn't exist
    dest_path = Path(dest_directory)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    supported_types = ['csv', 'xlsx', 'json', 'parquet', 'hdf', 'feather']
    if target_file_type not in supported_types:
        raise ValueError(f"Unsupported target file type. Supported types are: {supported_types}")

    for file_name in os.listdir(src_directory):
        file_path = os.path.join(src_directory, file_name)
        
        if os.path.isdir(file_path):
            logging.info(f"Skipping subdirectory: {file_path}")
            continue
        
        file_ext = file_name.split('.')[-1].lower()
        base_name = '.'.join(file_name.split('.')[:-1])
        
        if file_ext not in supported_types:
            logging.warning(f"Skipping unsupported file type: {file_path}")
            continue
        
        # Construct the destination file path
        dest_file_path = os.path.join(dest_directory, f"{base_name}.{target_file_type}")

        if file_ext == target_file_type:
            # Same file type, just copy
            shutil.copy(file_path, dest_file_path)
            logging.info(f"Copied {file_path} to {dest_file_path}")
        else:
            # Convert file to the target file type
            try:
                df = read_file(file_path, file_type=file_ext)
                if target_file_type == 'csv':
                    df.to_csv(dest_file_path, index=False)
                elif target_file_type == 'xlsx':
                    df.to_excel(dest_file_path, index=False)
                elif target_file_type == 'json':
                    df.to_json(dest_file_path)
                elif target_file_type == 'parquet':
                    df.to_parquet(dest_file_path)
                elif target_file_type == 'hdf':
                    df.to_hdf(dest_file_path, key='data', mode='w')
                elif target_file_type == 'feather':
                    df.to_feather(dest_file_path)
                logging.info(f"Converted {file_path} to {dest_file_path}")
            except Exception as e:
                logging.error(f"Error converting {file_path} to {target_file_type}: {e}")


def sample_data(df, n_samples=100):
    """
    Randomly sample rows from the DataFrame.
    
    Parameters:
    - df: DataFrame, the dataset.
    - n_samples: int, number of samples to take.
    
    Returns:
    - Sampled DataFrame.
    """
    try:
        sampled_df = df.sample(n=n_samples, random_state=1)
        logging.info(f"Sampled {n_samples} rows from the DataFrame.")
        return sampled_df
    except Exception as e:
        logging.error(f"Error during data sampling: {e}")
        raise

def plot_histogram(df, column):
    """
    Plot a histogram for a numeric column.
    
    Parameters:
    - df: DataFrame, the dataset.
    - column: str, the numeric column to plot.
    
    Returns:
    - None (displays a histogram plot).
    """
    try:
        df[column].hist()
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()
        logging.info(f"Histogram plotted for column '{column}'.")
    except Exception as e:
        logging.error(f"Error plotting histogram for column '{column}': {e}")
        raise


def full_clean(file_path, file_type='csv', force=False, numeric_fill_value=0, text_fill_value='null', match_columns=None, exact_match=True):
    """
    Perform a full cleaning of the dataset: check for duplicates, null values, and handle them.
    
    Parameters:
    - file_path: str, the path to the file.
    - file_type: str, the file type to read ('csv', 'xlsx', 'json', 'parquet', 'hdf', 'feather').
    - force: bool, whether to fill null values and delete duplicates (default is False).
    - numeric_fill_value: numeric, the value to fill in numeric columns (default is 0).
    - text_fill_value: str, the value to fill in text columns (default is 'null').
    - match_columns: list, specific columns to check for partial duplicates (default is None for exact matching).
    - exact_match: bool, whether to look for exact duplicates (default is True).
    
    Returns:
    - DataFrame cleaned if force=True, otherwise original DataFrame.
    """
    try:
        # Read the file
        df = read_file(file_path, file_type)
        
        logging.info("Starting full dataset cleaning process.")

        # Step 1: Check for duplicates
        logging.info("Checking for duplicates...")
        duplicates = check_duplicates(df, exact_match=exact_match, match_columns=match_columns, delete_duplicates=force)

        if not duplicates.empty:
            logging.warning(f"Found {len(duplicates)} duplicate rows.")
            if force:
                logging.info("Deleting duplicates...")
                df = check_duplicates(df, exact_match=exact_match, match_columns=match_columns, delete_duplicates=True)
            else:
                logging.info("Keeping duplicates since force is False.")
        else:
            logging.info("No duplicates found.")

        # Step 2: Check for null values
        logging.info("Checking for null values...")
        null_summary = check_nulls(df)

        if not null_summary.empty:
            logging.warning(f"Found null values in columns: {null_summary.index.tolist()}")
            if force:
                logging.info("Filling null values...")
                df = check_nulls(df, fill_null=True, numeric_fill_value=numeric_fill_value, text_fill_value=text_fill_value)
            else:
                logging.info("Keeping null values since force is False.")
        else:
            logging.info("No null values found.")

        # Step 3: Validate data types (optional step, based on the dataset)
        # Add any additional validation/cleaning steps here if needed.
        # For example, you can include type conversion or outlier detection.

        logging.info("Full cleaning completed.")

        # Step 4: Save the cleaned file if changes were made
        if force:
            cleaned_file_path = f"{file_path.split('.')[0]}_cleaned.{file_type}"
            logging.info(f"Saving cleaned dataset to {cleaned_file_path}")
            if file_type.lower() == 'csv':
                df.to_csv(cleaned_file_path, index=False)
            elif file_type.lower() == 'xlsx':
                df.to_excel(cleaned_file_path, index=False)
            elif file_type.lower() == 'json':
                df.to_json(cleaned_file_path)
            elif file_type.lower() == 'parquet':
                df.to_parquet(cleaned_file_path)
            elif file_type.lower() == 'hdf':
                df.to_hdf(cleaned_file_path, key='data', mode='w')
            elif file_type.lower() == 'feather':
                df.to_feather(cleaned_file_path)
            logging.info(f"Cleaned file saved successfully: {cleaned_file_path}")
        
        return df
    
    except Exception as e:
        logging.error(f"Error during full cleaning: {e}")
        raise

def mass_clean(src_directory, target_file_type, force=False, numeric_fill_value=0, text_fill_value='null', match_columns=None, exact_match=True, log_file_name='cleaning_log.txt'):
    """
    Perform mass cleaning of files in the source directory: convert them to a uniform file type, and clean them.
    
    Parameters:
    - src_directory: str, the path to the source directory containing files to process.
    - target_file_type: str, the desired file type for all files ('csv', 'xlsx', 'json', 'parquet', 'hdf', 'feather').
    - force: bool, whether to fill null values and delete duplicates (default is False).
    - numeric_fill_value: numeric, the value to fill in numeric columns (default is 0).
    - text_fill_value: str, the value to fill in text columns (default is 'null').
    - match_columns: list, specific columns to check for partial duplicates (default is None for exact matching).
    - exact_match: bool, whether to look for exact duplicates (default is True).
    - log_file_name: str, name of the log file to store the cleaning process logs.
    
    Returns:
    - None
    """
    # Set up logging
    log_directory = f"{src_directory}_cleaned"
    Path(log_directory).mkdir(parents=True, exist_ok=True)
    log_file_path = os.path.join(log_directory, log_file_name)
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info(f"Starting mass cleaning in directory '{src_directory}'.")

    try:
        # Step 1: Convert and copy files
        logging.info(f"Converting files to {target_file_type} and copying to '{log_directory}'.")
        convert_and_copy_files(src_directory, log_directory, target_file_type)
    except Exception as e:
        logging.error(f"Error during file conversion and copying: {e}")
        return  # Exit if conversion fails

    # Step 2: Clean the files
    logging.info(f"Starting cleaning process for files in '{log_directory}'.")
    for file_name in os.listdir(log_directory):
        file_path = os.path.join(log_directory, file_name)
        
        if os.path.isdir(file_path):
            logging.info(f"Skipping subdirectory: {file_path}")
            continue
        
        file_ext = file_name.split('.')[-1].lower()
        
        if file_ext != target_file_type:
            logging.warning(f"Skipping file with unexpected extension: {file_path}")
            continue
        
        try:
            logging.info(f"Processing file: {file_path}")
            full_clean(file_path, file_type=target_file_type, force=force, numeric_fill_value=numeric_fill_value, text_fill_value=text_fill_value, match_columns=match_columns, exact_match=exact_match)
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
        except pd.errors.EmptyDataError:
            logging.error(f"Empty data error for file: {file_path}")
        except pd.errors.ParserError:
            logging.error(f"Parser error for file: {file_path}")
        except Exception as e:
            logging.error(f"Error cleaning file {file_path}: {e}")

    logging.info("Mass cleaning process completed.")
    logging.info(f"Logs saved to {log_file_path}")