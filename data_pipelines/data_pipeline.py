import os
import sys
import pandas as pd
import yaml
from datetime import datetime
from typing import Dict, Tuple, List, Union, Optional
from pathlib import Path
from xbbg import blp
import logging
import numpy as np

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import utility functions
from src.utils.transformations import convert_er_ytd_to_index
from src.utils.data_merger import merge_dfs

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPipeline:
    def __init__(
        self,
        config_path: str = None,
        start_date: str = None,
        end_date: str = None,
        periodicity: str = None,
        align_start: bool = None,
        fill: str = None,
        start_date_align: str = None,
        ohlc_mapping: Dict[Tuple[str, str], str] = None,
        er_ytd_mapping: Dict[Tuple[str, str], str] = None,
        bad_dates: Dict[str, Dict[str, str]] = None
    ):
        """Initialize the DataPipeline class with configuration parameters."""
        # Load configuration from YAML if provided
        if config_path:
            self.load_config(config_path)
        else:
            # Default mappings
            DEFAULT_OHLC_MAPPING = {
                ('I05510CA Index', 'INDEX_OAS_TSY_BP'): 'cad_oas',
                ('LF98TRUU Index', 'INDEX_OAS_TSY_BP'): 'us_hy_oas',
                ('LUACTRUU Index', 'INDEX_OAS_TSY_BP'): 'us_ig_oas',
                ('SPTSX Index', 'PX_LAST'): 'tsx',
                ('VIX Index', 'PX_LAST'): 'vix',
                ('USYC3M30 Index', 'PX_LAST'): 'us_3m_10y',
                ('BCMPUSGR Index', 'PX_LAST'): 'us_growth_surprises',
                ('BCMPUSIF Index', 'PX_LAST'): 'us_inflation_surprises',
                ('LEI YOY  Index', 'PX_LAST'): 'us_lei_yoy',
                ('.HARDATA G Index', 'PX_LAST'): 'us_hard_data_surprises',
                ('CGERGLOB Index', 'PX_LAST'): 'us_equity_revisions',
                ('.ECONREGI G Index', 'PX_LAST'): 'us_economic_regime',
            }

            DEFAULT_ER_YTD_MAPPING = {
                ('I05510CA Index', 'INDEX_EXCESS_RETURN_YTD'): 'cad_ig_er',
                ('LF98TRUU Index', 'INDEX_EXCESS_RETURN_YTD'): 'us_hy_er',
                ('LUACTRUU Index', 'INDEX_EXCESS_RETURN_YTD'): 'us_ig_er',
            }

            DEFAULT_BAD_DATES = {
                '2005-11-15': {'column': 'cad_oas', 'action': 'use_previous'}
            }

            # Set parameters with defaults
            self.start_date = start_date or '2002-01-01'
            self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
            self.periodicity = periodicity or 'D'
            self.align_start = align_start if align_start is not None else True
            self.fill = fill or 'ffill'
            self.start_date_align = start_date_align or 'yes'
            self.ohlc_mapping = ohlc_mapping or DEFAULT_OHLC_MAPPING
            self.er_ytd_mapping = er_ytd_mapping or DEFAULT_ER_YTD_MAPPING
            self.bad_dates = bad_dates or DEFAULT_BAD_DATES

    def load_config(self, config_path: str):
        """Load configuration from a YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            pipeline_config = config.get('data_pipeline', {})
            
            self.start_date = pipeline_config.get('start_date', '2002-01-01')
            self.end_date = pipeline_config.get('end_date') or datetime.now().strftime('%Y-%m-%d')
            self.periodicity = pipeline_config.get('periodicity', 'D')
            self.align_start = pipeline_config.get('align_start', True)
            self.fill = pipeline_config.get('fill', 'ffill')
            self.start_date_align = pipeline_config.get('start_date_align', 'yes')
            
            # Load mappings
            mappings_config = pipeline_config.get('mappings', {})
            
            # Convert mapping strings to tuples
            self.ohlc_mapping = {}
            self.er_ytd_mapping = {}
            
            for mapping_str, column_name in mappings_config.get('ohlc_mapping', {}).items():
                ticker, field = mapping_str.split('|')
                self.ohlc_mapping[(ticker.strip(), field.strip())] = column_name
            
            for mapping_str, column_name in mappings_config.get('er_ytd_mapping', {}).items():
                ticker, field = mapping_str.split('|')
                self.er_ytd_mapping[(ticker.strip(), field.strip())] = column_name
            
            self.bad_dates = pipeline_config.get('bad_dates', {})
            
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {str(e)}")
            raise

    def update_parameters(self, **kwargs):
        """Update any of the class parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"DataPipeline has no attribute '{key}'")

        if 'end_date' in kwargs and kwargs['end_date'] is None:
            self.end_date = datetime.now().strftime('%Y-%m-%d')

    def fetch_bloomberg_data(self, mapping: Optional[Dict[Tuple[str, str], str]] = None) -> pd.DataFrame:
        """Fetch data from Bloomberg using xbbg."""
        try:
            mapping_to_use = mapping if mapping is not None else self.ohlc_mapping
            securities = list(set(security for security, _ in mapping_to_use.keys()))
            fields = list(set(field for _, field in mapping_to_use.keys()))

            logger.info(f"Fetching Bloomberg data for {len(securities)} securities")
            df = blp.bdh(
                tickers=securities,
                flds=fields,
                start_date=self.start_date,
                end_date=self.end_date,
                Per=self.periodicity
            )

            renamed_df = pd.DataFrame(index=df.index)
            for (security, field), new_name in mapping_to_use.items():
                if (security, field) in df.columns:
                    renamed_df[new_name] = df[(security, field)]
                else:
                    logger.warning(f"Column ({security}, {field}) not found in Bloomberg data")

            return renamed_df
        
        except Exception as e:
            logger.error(f"Error fetching Bloomberg data: {str(e)}")
            raise

    def convert_er_ytd_to_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert excess return YTD data to indices.
        
        The Bloomberg INDEX_EXCESS_RETURN_YTD field returns year-to-date percentage returns.
        This method converts these YTD returns into an index starting at 100 for each calendar year,
        where the index value for a given day is based on the previous year-end index value
        multiplied by (1 + YTD return/100).
        """
        try:
            result = pd.DataFrame(index=df.index)
            er_columns = list(self.er_ytd_mapping.values())
            
            # Extract year from index using datetime properties
            years = [date.year for date in df.index]
            df_years = pd.Series(years, index=df.index)
            
            for column in df.columns:
                if column in er_columns:
                    # Initialize the index at 100 for the starting date
                    index_values = pd.Series(np.nan, index=df.index)
                    prev_year_end = 100  # Start with 100
                    
                    # Process each year separately
                    for year in sorted(set(years)):
                        # Get data for this year
                        year_indices = [i for i, y in enumerate(years) if y == year]
                        if not year_indices:
                            continue
                            
                        # Get dates for this year
                        year_dates = df.index[year_indices]
                        if len(year_dates) == 0:
                            continue
                            
                        # Calculate index for each day in the year
                        for date in year_dates:
                            if pd.notna(df.loc[date, column]):
                                # Index = previous year-end value * (1 + YTD return/100)
                                index_values.loc[date] = prev_year_end * (1 + df.loc[date, column]/100)
                        
                        # Update prev_year_end for next year if we have data for the last day of the year
                        last_date_of_year = year_dates[-1]
                        if pd.notna(index_values.loc[last_date_of_year]):
                            prev_year_end = index_values.loc[last_date_of_year]
                    
                    # Fill forward any NaN values
                    index_values = index_values.fillna(method='ffill')
                    result[f"{column}_index"] = index_values
            
            return result
        
        except Exception as e:
            logger.error(f"Error converting excess returns to index: {str(e)}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean known data issues."""
        try:
            cleaned_df = df.copy()
            
            for date, info in self.bad_dates.items():
                if date in cleaned_df.index and info['column'] in cleaned_df.columns:
                    if info['action'] == 'use_previous':
                        prev_value = cleaned_df.loc[cleaned_df.index < date, info['column']].iloc[-1]
                        cleaned_df.loc[date, info['column']] = prev_value
                        logger.info(f"Cleaned bad data point for {info['column']} on {date}")
            
            return cleaned_df
        
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise

    def get_full_dataset(self) -> pd.DataFrame:
        """Get complete dataset with both price data and excess return indices."""
        try:
            logger.info("Fetching main price data...")
            df_ohlc = self.fetch_bloomberg_data(mapping=self.ohlc_mapping)
            
            logger.info("Fetching excess return YTD data...")
            er_ytd_df = self.fetch_bloomberg_data(mapping=self.er_ytd_mapping)
            
            logger.info("Converting excess returns to indices...")
            er_index_df = self.convert_er_ytd_to_index(er_ytd_df)
            
            logger.info("Merging datasets...")
            final_df = merge_dfs(df_ohlc, er_index_df, fill=self.fill, start_date_align=self.start_date_align)
            
            logger.info("Cleaning data...")
            final_df = self.clean_data(final_df)

            if self.start_date_align == 'yes':
                non_null_df = final_df.dropna(how='any')
                if not non_null_df.empty:
                    first_complete_date = non_null_df.index[0]
                    final_df = final_df[final_df.index >= first_complete_date]
                    logger.info(f"Aligned data to start from first complete date: {first_complete_date}")

            if self.fill:
                final_df = final_df.fillna(method=self.fill)
            
            return final_df
        
        except Exception as e:
            logger.error(f"Error getting full dataset: {str(e)}")
            raise

    def save_dataset(self, output_path=None, date_format=None):
        """Save the dataset to a file.
        
        Args:
            output_path (str): Path to save the dataset to.
            date_format (str): Format to use for the date index.
            
        Returns:
            str: The path to the saved dataset.
        """
        if output_path is None:
            output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_data.csv')
        
        # Get the dataset
        df = self.get_full_dataset()
        
        # Reset index to create Date column
        df_to_save = df.reset_index()
        df_to_save.rename(columns={'index': 'Date'}, inplace=True)
        
        # Save to CSV with Date as a column
        if date_format:
            df_to_save.to_csv(output_path, index=False, date_format=date_format)
        else:
            df_to_save.to_csv(output_path, index=False)
        
        logger.info(f"Dataset saved to {output_path}")
        return output_path

    @staticmethod
    def load_dataset(file_path, parse_dates=True):
        """Load a dataset from a file.
        
        Args:
            file_path (str): Path to the dataset.
            parse_dates (bool): Whether to parse the dates.
            
        Returns:
            pd.DataFrame: The loaded dataset.
        """
        if parse_dates:
            # Read with Date as a column
            df = pd.read_csv(file_path, parse_dates=['Date'])
            # Set Date as index
            df.set_index('Date', inplace=True)
        else:
            # Don't parse dates
            df = pd.read_csv(file_path)
                
        return df

if __name__ == '__main__':
    # Check if we're in an interactive environment by looking for specific args
    import sys
    is_interactive = any('ipykernel' in arg for arg in sys.argv) or '-f' in sys.argv or any('--f=' in arg for arg in sys.argv)
    
    if is_interactive:
        # When running in interactive window, just run with default settings
        try:
            print("Running data pipeline with default settings...")
            pipeline = DataPipeline()
            
            # Save CSV with Date as column
            output_path = pipeline.save_dataset()
            
            print(f"\nData pipeline completed successfully!")
            print(f"Dataset saved to: {output_path}")
            
            # Load and display detailed info about the data
            df = DataPipeline.load_dataset(output_path)
            
            print("\n" + "="*80)
            print("DATASET OVERVIEW")
            print("="*80)
            
            # Basic info
            print("\n[DATASET INFO]")
            print(f"Shape: {df.shape}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            print(f"Total days: {len(df)}")
            print(f"Total columns: {len(df.columns)}")
            
            # Detailed info
            print("\n[DETAILED INFO]")
            df.info()
            
            # Check for missing values
            print("\n[MISSING VALUE CHECK]")
            missing = df.isnull().sum()
            if missing.sum() > 0:
                print("Columns with missing values:")
                print(missing[missing > 0])
            else:
                print("No missing values found.")
            
            # Check for duplicated index values
            print("\n[DUPLICATE DATES CHECK]")
            duplicates = df.index.duplicated().sum()
            if duplicates > 0:
                print(f"WARNING: Found {duplicates} duplicate dates in the index!")
                print("Duplicate dates:")
                print(df[df.index.duplicated(keep=False)].index.tolist())
            else:
                print("No duplicate dates found.")
            
            # First 5 rows - display with Date as column for readability
            df_display = df.reset_index()
            df_display.rename(columns={'index': 'Date'}, inplace=True)
            
            print("\n[FIRST 5 ROWS]")
            print(df_display.head())
            
            # Last 5 rows
            print("\n[LAST 5 ROWS]")
            print(df_display.tail())
            
            # Statistical summary
            print("\n[STATISTICAL SUMMARY]")
            print(df.describe())
            
            # Check for unusual values (NaN, inf, extremely large/small)
            print("\n[DATA QUALITY CHECK]")
            inf_count = np.isinf(df).sum().sum()
            if inf_count > 0:
                print(f"WARNING: Found {inf_count} infinite values in the dataset!")
            
            # Check for ER index columns specifically
            er_cols = [col for col in df.columns if '_er_index' in col]
            if er_cols:
                print("\n[ER INDEX ANALYSIS]")
                for col in er_cols:
                    print(f"\n{col}:")
                    print(f"  Min: {df[col].min():.4f}")
                    print(f"  Max: {df[col].max():.4f}")
                    print(f"  Last: {df[col].iloc[-1]:.4f}")
                    print(f"  First: {df[col].iloc[0]:.4f}")
                    print(f"  % Change (first to last): {(df[col].iloc[-1]/df[col].iloc[0]-1)*100:.2f}%")
            
            # Available columns categorized
            print("\n[AVAILABLE COLUMNS BY TYPE]")
            er_cols = [col for col in df.columns if '_er_' in col]
            price_cols = [col for col in df.columns if '_price' in col]
            spread_cols = [col for col in df.columns if '_spread' in col or '_oas' in col]
            other_cols = [col for col in df.columns if col not in er_cols + price_cols + spread_cols]
            
            print("\nER Indices:")
            for col in er_cols:
                print(f"- {col}")
            
            print("\nPrice Columns:")
            for col in price_cols:
                print(f"- {col}")
            
            print("\nSpread/OAS Columns:")
            for col in spread_cols:
                print(f"- {col}")
            
            print("\nOther Columns:")
            for col in other_cols:
                print(f"- {col}")
            
        except Exception as e:
            logger.error(f"Error in interactive execution: {str(e)}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        # Normal command-line execution with arguments
        import argparse
        
        parser = argparse.ArgumentParser(description='Data Pipeline for Backtesting')
        parser.add_argument('--config', type=str, help='Path to configuration file')
        parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
        parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
        parser.add_argument('--periodicity', type=str, help='Data frequency (D, M)')
        parser.add_argument('--output', type=str, help='Output file path')
        parser.add_argument('--date-format', type=str, help='Format for datetime index in CSV')
        
        args = parser.parse_args()
        
        try:
            # Create the data pipeline
            pipeline = DataPipeline(config_path=args.config)
            
            # Update parameters if provided via command line
            update_params = {}
            if args.start_date:
                update_params['start_date'] = args.start_date
            if args.end_date:
                update_params['end_date'] = args.end_date
            if args.periodicity:
                update_params['periodicity'] = args.periodicity
            
            if update_params:
                pipeline.update_parameters(**update_params)
            
            # Get and save the data to CSV
            output_path = pipeline.save_dataset(args.output, date_format=args.date_format)
            
            # Load and display data info
            print(f"\nDataset Info:")
            df = DataPipeline.load_dataset(output_path)
            df_display = df.reset_index()
            df_display.rename(columns={'index': 'Date'}, inplace=True)
            
            print(df.info())
            print(f"\nFirst few rows of the data:")
            print(df_display.head())
            print(f"\nLast few rows of the data:")
            print(df_display.tail())
            print(f"\nData has been exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error in main execution: {str(e)}")
            sys.exit(1)
