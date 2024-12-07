"""
Bloomberg Data Pipeline

This script fetches, processes, and validates financial market data from Bloomberg across multiple indices
and data types. It handles various types of data including:
- Spreads (OAS indices)
- Derivatives (CDX indices)
- Excess Returns Year-to-Date

The script reads configuration from config.yaml, processes the data with proper handling of missing values,
and exports the results to CSV files with comprehensive validation and logging.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import sys
import yaml
import matplotlib.pyplot as plt
import logging.config
import json
from typing import Dict
from xbbg import blp

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import local modules
from src.utils.csv_exporter import export_table_to_csv, read_csv_to_df
from src.utils.validation import validate_dataframe, handle_missing_values

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

def setup_logging() -> None:
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

def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def print_dataframe_info(df: pd.DataFrame, name: str, validation_logger: logging.Logger) -> None:
    """Print comprehensive information about a DataFrame."""
    print(f"\n{name} Data after Bloomberg fetch:")
    print("-" * 50)
    print(f"Date Range: {df.index[0]} to {df.index[-1]}")
    df.info()

def print_dataframe_validation(df: pd.DataFrame, name: str) -> None:
    """Print comprehensive validation information about a DataFrame."""
    print(f"\n{'='*50}")
    print(f"Data Validation for: {name}")
    print(f"{'='*50}\n")
    
    # DataFrame info
    print(f"{name} Data Info:")
    print("----------------------------------")
    print(df.info())
    print()
    
    # Date range analysis
    print("Date Range:", df.index.min(), "to", df.index.max())
    
    # Check for missing business days
    business_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
    missing_days = business_days.difference(df.index)
    if len(missing_days) > 0:
        print(f"WARNING: Found {len(missing_days)} missing business days")
        print(f"First few missing dates: {list(missing_days[:5])}")
    print()
    
    # Data types
    print("Column Data Types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    print()
    
    # Value range validation
    print("Value Range Validation:\n")
    print("Basic Statistics:\n")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            stats = df[col].describe()
            last_value = df[col].iloc[-1]
            # Calculate percentile rank of last value
            percentile_rank = stats['count'] * (df[col] <= last_value).mean()
            percentile = (percentile_rank / stats['count']) * 100
            
            print(f"{col}:")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  Std: {stats['std']:.2f}")
            print(f"  Min: {stats['min']:.2f}")
            print(f"  Max: {stats['max']:.2f}")
            print(f"  Last Value: {last_value:.2f}")
            print(f"  Percentile Rank: {percentile:.1f}%")
            print()
    
    # First and last rows
    print("\nFirst 10 rows:")
    print("----------------------------------")
    print(df.head(10))
    print("\nLast 10 rows:")
    print("----------------------------------")
    print(df.tail(10))
    print()

def plot_historical_data(df: pd.DataFrame, name: str, output_dir: Path, data_pipeline_logger: logging.Logger) -> None:
    """Create and save historical line charts for all numeric columns."""
    plots_dir = output_dir / 'plots' / name.lower()
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots for each numeric column
    for col in df.select_dtypes(include=[np.number]).columns:
        plt.figure()
        df[col].plot(title=f'{col} Historical Data')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.tight_layout()
        
        # Sanitize filename by replacing invalid characters
        safe_col = col.replace('>', 'gt').replace('<', 'lt').replace(':', '_')
        plot_path = plots_dir / f'{safe_col}_historical.png'
        
        plt.savefig(plot_path)
        plt.close()
        
        data_pipeline_logger.info(f"Created plot for {col} at {plot_path}")

def fetch_bloomberg_data() -> Dict[str, pd.DataFrame]:
    """Fetch data from Bloomberg for all configured sections."""
    # Load configuration
    config = load_config()
    
    # Get start date from config settings
    start_date = config['settings']['default_start_date']
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Dictionary to store DataFrames for each section
    section_dfs = {}
    
    # Process each section (sprds, derv, er_ytd)
    for section_name in ['sprds', 'derv', 'er_ytd']:
        if section_name in config:
            section = config[section_name]
            df = pd.DataFrame()
            
            for security in section['securities']:
                data = blp.bdh(security['ticker'], section['field'], 
                          start_date, end_date,
                          cache=False)  # Disable caching
                if not data.empty:
                    # Access the data using multi-index column structure
                    series = data[(security['ticker'], section['field'])]
                    
                    # Replace 0s with NaN before first valid data point
                    first_valid_idx = series.first_valid_index()
                    if first_valid_idx is not None:
                        # Set all values before first valid to NaN (including 0s)
                        series.loc[:first_valid_idx] = np.nan
                        # Forward fill after first valid point, replacing 0s
                        mask = series.index >= first_valid_idx
                        series.loc[mask] = series.loc[mask].replace(0, np.nan).ffill()
                    
                    # For Sprds data, handle negative values by replacing with previous valid value
                    if section_name == 'sprds':
                        # Create a mask for negative values
                        negative_mask = series < 0
                        if negative_mask.any():
                            # Replace negative values with NaN
                            series[negative_mask] = np.nan
                            # Forward fill to replace NaNs with previous valid values
                            series = series.ffill()
                    
                    # Special handling for Nov 15, 2005
                    problem_date = pd.Timestamp('2005-11-15')
                    if problem_date in series.index:
                        # Get the previous day's value
                        prev_date = series.index[series.index.get_loc(problem_date) - 1]
                        series.loc[problem_date] = series.loc[prev_date]
                    
                    df[security['custom_name']] = series
            
            section_dfs[section_name] = df
            
    return section_dfs

def main():
    """Main function to run the Bloomberg data pipeline."""
    # Set up logging
    setup_logging()
    data_pipeline_logger = logging.getLogger('data_pipeline')
    validation_logger = logging.getLogger('validation')
    
    # Load configuration
    config = load_config()
    
    # Create output directories if they don't exist
    data_dir = project_root / config['settings']['data_directory']
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch data from Bloomberg
    section_dfs = fetch_bloomberg_data()
    
    # Process each section
    for section_name, df in section_dfs.items():
        if not df.empty:
            print(f"\n{'#'*80}")
            print(f"Processing {section_name.upper()} Data")
            print(f"{'#'*80}\n")
            
            # Print validation for data after Bloomberg fetch
            print("\nSpreads Data after Bloomberg fetch:")
            print("----------------------------------")
            print_dataframe_validation(df, section_name.title())
            
            # Export to CSV
            csv_path = data_dir / f'{section_name}_data.csv'
            df.to_csv(csv_path)
            data_pipeline_logger.info(f"Exported {section_name} data to {csv_path}")
            
            # Read back and validate
            df_from_csv = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            print("\nSpreads Data after reading from CSV:")
            print("----------------------------------")
            print_dataframe_validation(df_from_csv, section_name.title())
            
            # Create plots
            plot_historical_data(df_from_csv, section_name, project_root, data_pipeline_logger)

if __name__ == "__main__":
    main()
