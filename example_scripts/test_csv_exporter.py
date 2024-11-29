import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import logging
import logging.config
import json

# Set up plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = [15, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['lines.linewidth'] = 1.5

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.csv_exporter import export_table_to_csv, export_tables_to_csv
from src.core.bloomberg_fetcher import BloombergDataFetcher

def setup_logging():
    """Set up logging configuration."""
    config_path = project_root / 'config' / 'logging_config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Create loggers for different components
    data_pipeline_logger = logging.getLogger('data_pipeline')
    bloomberg_logger = logging.getLogger('bloomberg')
    validation_logger = logging.getLogger('validation')
    
    return data_pipeline_logger, bloomberg_logger, validation_logger

def load_config():
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
        
        plot_path = os.path.join(output_dir, f"{name}_historical.png")
        plt.savefig(plot_path)
        plt.close()
        
        data_pipeline_logger.info(f"Historical chart saved to: {plot_path}")
    except Exception as e:
        data_pipeline_logger.error(f"Error creating historical chart for {name}: {str(e)}")
        raise

def fetch_bloomberg_data(config, bloomberg_logger):
    """Fetch Bloomberg data based on config."""
    bloomberg_logger.info("Starting Bloomberg data fetch")
    bloomberg_logger.info(f"Configuration: {json.dumps(config['settings'], indent=2)}")
    
    try:
        # Create Bloomberg fetcher with config
        fetcher = BloombergDataFetcher(
            tickers=[],  # Empty because we're using config-based approach
            fields=[],   # Empty because we're using config-based approach
            start_date=config['settings']['default_start_date'],
            end_date=config['settings']['default_end_date'],
            logger=bloomberg_logger,
            config=config
        )
        
        # Fetch and process data
        data = fetcher.run_pipeline()
        bloomberg_logger.info("Bloomberg data fetch completed successfully")
        return data
    except Exception as e:
        bloomberg_logger.error(f"Error fetching Bloomberg data: {str(e)}")
        raise

def main():
    # Set up logging
    data_pipeline_logger, bloomberg_logger, validation_logger = setup_logging()
    data_pipeline_logger.info("Starting data export pipeline")
    
    try:
        # Load configuration
        config = load_config()
        data_pipeline_logger.info(f"Configuration loaded: {json.dumps(config['settings'], indent=2)}")
        
        # Create output directory for data and plots
        output_dir = project_root / config['settings']['data_directory'] / 'test_exports'
        output_dir.mkdir(parents=True, exist_ok=True)
        data_pipeline_logger.info(f"Output directory created: {output_dir}")
        
        # Fetch Bloomberg data
        data_pipeline_logger.info("Initiating Bloomberg data fetch")
        market_data = fetch_bloomberg_data(config, bloomberg_logger)
        
        # Split data into spreads and derivatives
        spreads_cols = ['cad_ig_oas', 'us_ig_oas']
        derv_cols = ['cdx_ig', 'cdx_hy']
        
        spreads_data = market_data[spreads_cols].copy()
        derivatives_data = market_data[derv_cols].copy()
        
        # Add Date column for CSV export
        spreads_data = spreads_data.reset_index()
        derivatives_data = derivatives_data.reset_index()
        
        # Validate data
        validation_logger.info("Starting data validation")
        print_dataframe_info(spreads_data, "Spreads Data", validation_logger)
        print_dataframe_info(derivatives_data, "Derivatives Data", validation_logger)
        
        # Create visualizations
        data_pipeline_logger.info("Creating visualizations")
        plot_historical_data(spreads_data.set_index('index'), "spreads_data", output_dir, data_pipeline_logger)
        plot_historical_data(derivatives_data.set_index('index'), "derivatives_data", output_dir, data_pipeline_logger)
        
        # Export data to CSV
        data_pipeline_logger.info("Exporting data to CSV")
        spreads_file = export_table_to_csv(spreads_data, "spreads_data_data", output_dir)
        derivatives_file = export_table_to_csv(derivatives_data, "derivatives_data_data", output_dir)
        
        data_pipeline_logger.info(f"Data exported successfully to:\n- {spreads_file}\n- {derivatives_file}")
        
        # Verify exported data
        validation_logger.info("Verifying exported data")
        for file_path in [spreads_file, derivatives_file]:
            try:
                loaded_df = pd.read_csv(file_path)
                validation_logger.info(f"Successfully loaded: {Path(file_path).name}")
                validation_logger.info(f"Shape verification: {loaded_df.shape}")
                validation_logger.info(f"Columns: {loaded_df.columns.tolist()}")
                validation_logger.info(f"Data Types:\n{loaded_df.dtypes}")
            except Exception as e:
                validation_logger.error(f"Error verifying {Path(file_path).name}: {str(e)}")
        
        data_pipeline_logger.info("Data pipeline completed successfully")
        
    except Exception as e:
        data_pipeline_logger.error(f"Error in data pipeline: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
