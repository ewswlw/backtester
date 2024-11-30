import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Now import our local modules
from src.utils.csv_exporter import export_table_to_csv, read_csv_to_df
from src.utils.validation import validate_dataframe, handle_missing_values

import yaml
import matplotlib.pyplot as plt
import logging.config
import json
from typing import Tuple
from xbbg import blp
import sys
from datetime import datetime, timedelta

# Set up plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = [15, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['lines.linewidth'] = 1.5

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_logging():
    """Set up logging configuration."""
    config_path = project_root / 'config' / 'logging_config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            # Modify log file paths to use absolute paths
            for handler in config['handlers'].values():
                if 'filename' in handler:
                    handler['filename'] = str(project_root / 'logs' / Path(handler['filename']).name)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Create logs directory if it doesn't exist
    logs_dir = project_root / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create loggers for different components
    data_pipeline_logger = logging.getLogger('data_pipeline')
    bloomberg_logger = logging.getLogger('bloomberg')
    validation_logger = logging.getLogger('validation')
    
    return data_pipeline_logger, bloomberg_logger, validation_logger

def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def print_dataframe_info(df, name, validation_logger):
    """Print comprehensive information about a DataFrame."""
    validation_logger.info(f"Analyzing DataFrame: {name}")
    
    # Log basic information
    validation_logger.info(f"Shape: {df.shape}")
    validation_logger.info(f"Memory Usage: {df.memory_usage().sum() / 1024:.2f} KB")
    
    # Log data types
    validation_logger.info(f"Data Types:\n{df.dtypes}")
    
    # Log summary statistics
    stats = df.describe(include='all').round(4)
    validation_logger.info(f"Summary Statistics:\n{stats}")
    
    # Log missing values
    missing = df.isnull().sum()
    if missing.any():
        validation_logger.warning(f"Missing Values Found:\n{missing[missing > 0]}")
    else:
        validation_logger.info("No missing values found")
    
    # Log data ranges for numeric columns
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            stats = {
                'min': df[col].min(),
                'max': df[col].max(),
                'range': df[col].max() - df[col].min(),
                'std_dev': df[col].std()
            }
            validation_logger.info(f"Column {col} statistics: {json.dumps(stats, indent=2)}")

def print_dataframe_validation(df, name):
    """Print comprehensive validation information about a DataFrame."""
    print(f"\n{'='*50}")
    print(f"Data Validation for: {name}")
    print(f"{'='*50}")
    
    # Basic info
    print("\nDataFrame Info:")
    print("-" * 20)
    df.info()
    
    # First and last rows
    print("\nFirst 10 rows:")
    print("-" * 20)
    print(df.head(10))
    
    print("\nLast 10 rows:")
    print("-" * 20)
    print(df.tail(10))
    
    # Statistical description
    print("\nDescriptive Statistics:")
    print("-" * 20)
    print(df.describe())
    
    # Missing values check
    missing = df.isnull().sum()
    if missing.any():
        print("\nMissing Values:")
        print("-" * 20)
        print(missing[missing > 0])
    
    # Duplicates check
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"\nWarning: Found {duplicates} duplicate rows")
    
    # Data range check
    print("\nData Range for each column:")
    print("-" * 20)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"{col}:")
            print(f"  Min: {df[col].min()}")
            print(f"  Max: {df[col].max()}")
            print(f"  Range: {df[col].max() - df[col].min()}")
    
    print("\nUnique values count per column:")
    print("-" * 20)
    print(df.nunique())

def plot_historical_data(df, name, output_dir, data_pipeline_logger):
    """Create and save historical line charts for all numeric columns."""
    data_pipeline_logger.info(f"Creating historical chart for {name}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        data_pipeline_logger.warning(f"No numeric columns found in {name}")
        return
    
    try:
        plt.figure(figsize=(12, 6), dpi=300)
        for col in numeric_cols:
            plt.plot(df.index, df[col], label=col, linewidth=1)
        
        plt.title(f'{name} Historical Data')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Use consistent filename without timestamp
        plot_path = os.path.join(output_dir, f"{name}_historical.png")
        plt.savefig(plot_path)
        plt.close()
        
        data_pipeline_logger.info(f"Historical chart saved to: {plot_path}")
        return plot_path
    except Exception as e:
        data_pipeline_logger.error(f"Error creating historical chart for {name}: {str(e)}")
        raise

def fetch_bloomberg_data(start_date: str = '2020-01-02') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch data from Bloomberg."""
    from xbbg import blp
    
    # Force cache refresh by adding a timestamp parameter
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    # Fetch spreads data
    spreads_tickers = {
        'I05510CA Index': 'cad_ig_oas',  # CAD Investment Grade OAS
        'LUACTRUU Index': 'us_ig_oas'    # US Investment Grade OAS
    }
    
    spreads_df = pd.DataFrame()
    for ticker, col_name in spreads_tickers.items():
        data = blp.bdh(ticker, 'INDEX_OAS_TSY_BP', start_date, datetime.now().strftime('%Y-%m-%d'),
                      cache=False)  # Disable caching
        if not data.empty:
            # Access the data using multi-index column structure
            spreads_df[col_name] = data[(ticker, 'INDEX_OAS_TSY_BP')]
    
    # Fetch derivatives data
    derivatives_tickers = {
        'IBOXUMAE MKIT Curncy': 'cdx_ig',  # CDX IG
        'IBOXHYSE MKIT Curncy': 'cdx_hy'   # CDX HY
    }
    
    derivatives_df = pd.DataFrame()
    for ticker, col_name in derivatives_tickers.items():
        data = blp.bdh(ticker, 'ROLL_ADJUSTED_MID_PRICE', start_date, datetime.now().strftime('%Y-%m-%d'),
                      cache=False)  # Disable caching
        if not data.empty:
            # Access the data using multi-index column structure
            derivatives_df[col_name] = data[(ticker, 'ROLL_ADJUSTED_MID_PRICE')]
    
    return spreads_df, derivatives_df

def read_csv_to_df(file_path: Path, date_column: str = 'Date') -> pd.DataFrame:
    """
    Read a CSV file into a DataFrame and set the date column as the index.
    
    Args:
        file_path (Path): Path to the CSV file
        date_column (str): Name of the date column to use as index. Defaults to 'Date'
    
    Returns:
        pd.DataFrame: DataFrame with DatetimeIndex
    """
    # Read CSV
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime and set as index
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
    
    return df

def main():
    """Main function to run the example."""
    # Set up logging
    data_pipeline_logger, bloomberg_logger, validation_logger = setup_logging()
    
    try:
        # Load configuration
        config = load_config()
        data_pipeline_logger.info("Configuration loaded successfully")
        
        # Create output directories
        data_dir = project_root / config['settings']['data_directory']
        plots_dir = project_root / 'plots'
        data_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Fetch data from Bloomberg
        spreads_data, derivatives_data = fetch_bloomberg_data()
        data_pipeline_logger.info("Bloomberg data fetched successfully")
        
        print("\nSpreads Data after Bloomberg fetch:")
        print("----------------------------------")
        spreads_data.info()
        
        print("\nDerivatives Data after Bloomberg fetch:")
        print("----------------------------------")
        derivatives_data.info()
        
        # Export to CSV
        spreads_output = data_dir / 'spreads_data.csv'
        derivatives_output = data_dir / 'derivatives_data.csv'
        
        export_table_to_csv(spreads_data, spreads_output)
        export_table_to_csv(derivatives_data, derivatives_output)
        data_pipeline_logger.info("Data exported to CSV successfully")
        
        # Read back and validate
        spreads_df = read_csv_to_df(spreads_output)
        derivatives_df = read_csv_to_df(derivatives_output)
        
        print("\nSpreads Data after reading from CSV:")
        print("----------------------------------")
        spreads_df.info()
        
        print("\nDerivatives Data after reading from CSV:")
        print("----------------------------------")
        derivatives_df.info()
        
        # Handle missing values in derivatives data
        derivatives_df_clean = handle_missing_values(derivatives_df, method='interpolate')
        
        # Validate the data
        spreads_insights = validate_dataframe(
            spreads_df, 
            "Spreads Data",
            expected_cols=['cad_ig_oas', 'us_ig_oas'],
            value_ranges={'cad_ig_oas': (50, 500), 'us_ig_oas': (50, 500)}
        )
        print("\n".join(spreads_insights))
        
        derivatives_insights = validate_dataframe(
            derivatives_df_clean,
            "Derivatives Data",
            expected_cols=['cdx_ig', 'cdx_hy'],
            value_ranges={'cdx_ig': (30, 200), 'cdx_hy': (250, 1000)}
        )
        print("\n".join(derivatives_insights))
        
        # Plot the data
        plot_historical_data(spreads_df, 'spreads', plots_dir, data_pipeline_logger)
        plot_historical_data(derivatives_df_clean, 'derivatives', plots_dir, data_pipeline_logger)
        data_pipeline_logger.info("Charts generated successfully")
        
    except Exception as e:
        data_pipeline_logger.error(f"Error in main pipeline: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
