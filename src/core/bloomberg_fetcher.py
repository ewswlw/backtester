import pandas as pd
from xbbg import blp
from typing import List, Dict, Optional
from pathlib import Path
import logging

class BloombergDataFetcher:
    """Class to fetch and process Bloomberg data."""
    
    def __init__(
        self,
        tickers: List[str],
        fields: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict] = None
    ):
        self.tickers = tickers
        self.fields = fields
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else pd.Timestamp.today()
        self.logger = logger or logging.getLogger(__name__)
        self.custom_names = {}
        self.config = config or {}
        
    def set_custom_column_names(self, custom_names: Dict[str, str]):
        """Set custom names for renaming columns."""
        self.custom_names = custom_names
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch data from Bloomberg."""
        try:
            all_data = []
            self.logger.info(f"Starting data fetch with tickers: {self.tickers} and fields: {self.fields}")
            
            # If using direct ticker/field approach
            if self.tickers and self.fields:
                for ticker in self.tickers:
                    for field in self.fields:
                        self.logger.info(f"Fetching data for ticker: {ticker}, field: {field}")
                        data = blp.bdh(
                            ticker,
                            field,
                            self.start_date,
                            self.end_date
                        )
                        
                        if not data.empty:
                            # Get the custom name for this ticker if available
                            custom_name = self.custom_names.get(ticker, ticker)
                            self.logger.info(f"Data fetched successfully for {ticker}, using custom name: {custom_name}")
                            
                            # Extract the data values and create a new DataFrame with the custom name
                            if isinstance(data.columns, pd.MultiIndex):
                                values = data.iloc[:, 0].values  # Get first column's values
                                data = pd.DataFrame(values, index=data.index, columns=[custom_name])
                            else:
                                data.columns = [custom_name]
                            
                            # Remove any rows with NaN values
                            data = data.dropna()
                            all_data.append(data)
                        else:
                            self.logger.warning(f"Empty data returned for ticker: {ticker}, field: {field}")
            
            # If using config-based approach
            elif self.config:
                for data_type, config in self.config.items():
                    if data_type not in ['sprds', 'derv', 'settings']:
                        continue
                        
                    if data_type in ['sprds', 'derv']:
                        field = config['field']
                        for security in config['securities']:
                            ticker = security['ticker']
                            custom_name = security['custom_name']
                            
                            self.logger.info(f"Fetching data for ticker: {ticker}, field: {field}")
                            data = blp.bdh(
                                ticker,
                                field,
                                self.start_date,
                                self.end_date
                            )
                            
                            if not data.empty:
                                self.logger.info(f"Data fetched successfully for {ticker}, using custom name: {custom_name}")
                                
                                # Extract the data values and create a new DataFrame with the custom name
                                if isinstance(data.columns, pd.MultiIndex):
                                    values = data.iloc[:, 0].values  # Get first column's values
                                    data = pd.DataFrame(values, index=data.index, columns=[custom_name])
                                else:
                                    data.columns = [custom_name]
                                
                                # Remove any rows with NaN values
                                data = data.dropna()
                                all_data.append(data)
                            else:
                                self.logger.warning(f"Empty data returned for ticker: {ticker}, field: {field}")
            
            if not all_data:
                self.logger.error("No data was fetched from Bloomberg")
                raise ValueError("No data fetched from Bloomberg")
            
            # Combine all data
            combined_data = pd.concat(all_data, axis=1)
            self.logger.info(f"Successfully combined data with columns: {combined_data.columns.tolist()}")
            
            # Remove any rows where any values are NaN
            combined_data = combined_data.dropna(how='any')
            
            # Sort index
            combined_data = combined_data.sort_index()
            
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            raise

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the fetched data."""
        try:
            # Remove future dates
            today = pd.Timestamp.today().normalize()  # Get today's date at midnight
            data = data[data.index.to_series().apply(lambda x: pd.Timestamp(x) <= today)]
            
            # Remove any rows where any values are NaN
            data = data.dropna(how='any')
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            raise
            
    def run_pipeline(self) -> pd.DataFrame:
        """Run the complete data fetching and processing pipeline."""
        self.logger.info("Starting data pipeline")
        self.logger.info(f"Using tickers: {self.tickers}")
        self.logger.info(f"Using fields: {self.fields}")
        self.logger.info(f"Custom names mapping: {self.custom_names}")
        
        data = self.fetch_data()
        self.logger.info("Data fetched successfully")
        
        processed_data = self.process_data(data)
        self.logger.info("Data processed successfully")
        
        return processed_data
