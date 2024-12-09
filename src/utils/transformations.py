import pandas as pd
from typing import List, Optional, Union
import logging
from xbbg import blp
import os

# Set up logging only if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_frequency(frequency: str) -> str:
    """Normalize frequency input to the expected Bloomberg format."""
    freq_map = {
        'DAILY': 'DAILY', 'D': 'DAILY', '1D': 'DAILY', 'DAY': 'DAILY',
        'WEEKLY': 'WEEKLY', 'W': 'WEEKLY', '1W': 'WEEKLY', 'WEEK': 'WEEKLY',
        'MONTHLY': 'MONTHLY', 'M': 'MONTHLY', '1M': 'MONTHLY', 'MONTH': 'MONTHLY',
        'QUARTERLY': 'QUARTERLY', 'Q': 'QUARTERLY', '1Q': 'QUARTERLY', 'QUARTER': 'QUARTERLY',
        'YEARLY': 'YEARLY', 'Y': 'YEARLY', '1Y': 'YEARLY', 'YEAR': 'YEARLY',
        'BM': 'MONTHLY', 'BME': 'MONTHLY', 'BMONTHLY': 'MONTHLY',  # Treat business month-end as monthly
        'ME': 'MONTHLY'  # Treat month-end as monthly
    }
    normalized_freq = freq_map.get(frequency.upper().strip(), 'DAILY')  # Default to 'DAILY' if not found
    return normalized_freq

def risk_free_index(df: pd.DataFrame, col_name: str = "risk_free") -> pd.DataFrame:
    """
    Computes an index starting at 100 that mimics the compounded effect of investing in the index with annual compounding.
    
    :param df: DataFrame with datetime index and interest rates in percentage format (e.g., 4.65 means 4.65%).
    :param col_name: Optional string to set the name of the compounded index column. Default is 'risk_free'.
    :return: DataFrame with the compounded index.
    """
    # Frequency conversion factors
    frequency_factors = {
        'D': 365,     # Daily
        'B': 252,     # Business day
        'M': 12,      # Monthly
        'Q': 4,       # Quarterly
        'A': 1,       # Annual
        'ME': 12      # Month-End, treated as monthly
    }
    
    # Infer the frequency of the DataFrame
    inferred_freq = pd.infer_freq(df.index)
    
    if inferred_freq is None:
        # Attempt to resample to the most likely frequency (daily) to fill gaps
        logging.warning("Cannot infer frequency. Attempting to resample to daily frequency.")
        df = df.resample('D').ffill()
        inferred_freq = pd.infer_freq(df.index)
        if inferred_freq is None:
            raise ValueError("Cannot infer frequency of the DataFrame index after resampling.")
    
    logging.info(f"Inferred frequency: {inferred_freq}")
    
    # Map the inferred frequency to the relevant period per year
    frequency_key = inferred_freq[:2]  # Use the first two characters to capture 'ME' and others
    if frequency_key not in frequency_factors:
        raise ValueError(f"Inferred frequency '{frequency_key}' not supported. Supported frequencies: 'D', 'B', 'M', 'Q', 'A', 'ME'.")
    
    periods_per_year = frequency_factors[frequency_key]
    
    # Flatten multi-index columns if present
    df.columns = [col[1] if isinstance(col, tuple) else col for col in df.columns]
    
    # Convert interest rates from percentage to decimal
    df = df / 100.0
    
    # Compute the compounded index
    compounded_index = (1 + df / periods_per_year).cumprod()
    
    # Normalize to start at 100
    compounded_index = 100 * compounded_index / compounded_index.iloc[0]
    
    # Rename the column to indicate it is an index
    compounded_index.columns = [col_name + '_index' for col in df.columns]
    
    return compounded_index

def get_credit_fund_estimated_index(df1: pd.DataFrame, df2: pd.DataFrame, col_name: str = "credit_fund_estimated_index") -> pd.DataFrame:
  """
  Computes an estimated credit fund index starting at 100 by summing the periodic returns of the two input DataFrames
  and constructing a single index. This function can handle DataFrames with different date ranges.

  :param df1: First DataFrame with datetime index and price levels (e.g., daily closing prices).
  :param df2: Second DataFrame with datetime index and price levels.
  :param col_name: Optional string to set the name of the compounded index column. Default is 'credit_fund_estimated_index'.
  :return: DataFrame with a single column representing the credit fund estimated index.
  """
  # Combine the indexes of both DataFrames
  combined_index = df1.index.union(df2.index)
  
  # Reindex both DataFrames to the combined index, forward-filling missing values
  df1_reindexed = df1.reindex(combined_index, method='ffill')
  df2_reindexed = df2.reindex(combined_index, method='ffill')
  
  # Calculate the periodic return of each index
  periodic_return1 = df1_reindexed.pct_change().fillna(0)
  periodic_return2 = df2_reindexed.pct_change().fillna(0)
  
  # Sum the returns from both DataFrames
  total_returns = periodic_return1.sum(axis=1) + periodic_return2.sum(axis=1)
  
  # Compute the estimated index
  estimated_index = (1 + total_returns).cumprod()
  
  # Normalize to start at 100
  estimated_index = 100 * estimated_index / estimated_index.iloc[0]
  
  # Convert the result to a DataFrame and name the column
  estimated_index = estimated_index.to_frame(name=col_name)
  
  return estimated_index


def get_er_index(df: pd.DataFrame, cost_of_borrow: float = 40, leverage: float = 1.0, col_name: str = "er_index") -> pd.DataFrame:
    """
    Computes an excess return index starting at 100 that represents the performance of an investment in the index,
    adjusted for the cost of borrowing and leverage.

    :param df: DataFrame with datetime index and price levels (e.g., daily closing prices).
    :param cost_of_borrow: A float expressed in basis points (e.g., 40 means 40 bps, or 0.40% annual compounding interest rate). Default is 40.
    :param leverage: A float that represents the leverage used to create the new index. Default is 1.0.
    :param col_name: Optional string to set the name of the compounded index column. Default is 'er_index'.
    :return: DataFrame with the excess return index.
    """
    # Convert cost_of_borrow from basis points to a decimal (e.g., 40 bps = 0.004)
    cost_of_borrow_decimal = cost_of_borrow / 10000
    
    # Infer the frequency of the DataFrame
    inferred_freq = pd.infer_freq(df.index)
    
    if inferred_freq is None:
        # Attempt to resample to the most likely frequency (daily) to fill gaps
        logging.warning("Cannot infer frequency. Attempting to resample to daily frequency.")
        df = df.resample('D').ffill()
        inferred_freq = pd.infer_freq(df.index)
        if inferred_freq is None:
            raise ValueError("Cannot infer frequency of the DataFrame index after resampling.")
    
    logging.info(f"Inferred frequency: {inferred_freq}")
    
    # Calculate daily cost of borrowing
    daily_cost = cost_of_borrow_decimal / 365
    
    # Calculate returns for each column
    result_df = pd.DataFrame(index=df.index)
    
    # Create leverage suffix using the actual leverage value
    leverage_suffix = f"_{leverage:.1f}x"
    
    for column in df.columns:
        # Calculate daily returns
        daily_returns = df[column].pct_change().fillna(0)
        
        # Apply leverage and subtract borrowing cost
        leveraged_returns = (daily_returns * leverage) - (daily_cost * leverage)
        
        # Calculate cumulative index
        cumulative_index = (1 + leveraged_returns).cumprod()
        
        # Normalize to start at 100
        normalized_index = 100 * cumulative_index / cumulative_index.iloc[0]
        
        # Add to result DataFrame with original column name plus leverage suffix
        result_df[f"{column}{leverage_suffix}"] = normalized_index
    
    return result_df

def convert_er_ytd_to_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts year-to-date excess return data in all columns to custom indices starting at 100.
    Assumes that all columns contain year-to-date excess return data in percentage format.

    :param df: DataFrame containing year-to-date excess return data in percentage format.
    :return: DataFrame with custom indices columns calculated from daily returns.
    """

    # 1. Ensure the DataFrame has a datetime index
    def ensure_datetime_index(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Ensures that the DataFrame has a datetime index. Converts 'Date' column to index if necessary."""
        if 'Date' in dataframe.columns:
            dataframe['Date'] = pd.to_datetime(dataframe['Date'], errors='coerce')
            dataframe.set_index('Date', inplace=True)
        if not isinstance(dataframe.index, pd.DatetimeIndex):
            raise ValueError("The DataFrame must have a datetime index or a 'Date' column.")
        return dataframe

    # 2. Convert returns from percentage to decimal
    def convert_returns_format(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Converts all columns from percentage to decimal format (divide by 100)."""
        dataframe = dataframe / 100
        return dataframe

    # 3 & 4. Calculate daily returns from YTD returns
    def calculate_daily_returns(ytd_returns: pd.Series) -> pd.Series:
        """Calculates daily returns from YTD returns for each year."""
        daily_returns = pd.Series(index=ytd_returns.index, dtype=float)
        
        for year in ytd_returns.index.year.unique():
            year_data = ytd_returns[ytd_returns.index.year == year]
            
            # Handle first day of the year
            daily_returns.loc[year_data.index[0]] = year_data.iloc[0]
            
            # Calculate daily returns for the rest of the year
            daily_returns.loc[year_data.index[1:]] = (1 + year_data.iloc[1:].values) / (1 + year_data.iloc[:-1].values) - 1
        
        return daily_returns

    # 5. Create custom indices starting at 100
    def create_custom_index(daily_returns: pd.Series, start_value: float = 100) -> pd.Series:
        """Creates a custom index starting at a specified value from daily returns."""
        custom_index = start_value * (1 + daily_returns).cumprod()
        return custom_index

    # Process the DataFrame
    df = ensure_datetime_index(df)
    df = convert_returns_format(df)

    index_columns = {}

    # Calculate daily returns and create custom indices for each column
    for column in df.columns:
        daily_returns = calculate_daily_returns(df[column])
        index_column = create_custom_index(daily_returns)
        index_columns[f"{column}_index"] = index_column

    # Create a new DataFrame with the custom indices
    index_df = pd.DataFrame(index_columns, index=df.index)

    return index_df



def get_ohlc(data, frequency):
  """
  Resample the input time series data to the specified frequency and return OHLC values.

  Parameters:
  - data: pd.Series or pd.DataFrame
      The input time series data. If a DataFrame, it should have a DateTime index and a single column.
  - frequency: str
      The resampling frequency ('d', 'w', 'm', 'q', 'y').

  Returns:
  - pd.DataFrame
      A DataFrame with columns ['Open', 'High', 'Low', 'Close'].
  """
  # Ensure the input is a DataFrame
  if isinstance(data, pd.Series):
      data = data.to_frame(name='value')
  elif isinstance(data, pd.DataFrame):
      if data.shape[1] != 1:
          raise ValueError("DataFrame must have exactly one column.")
  else:
      raise TypeError("Input must be a Pandas Series or DataFrame.")

  # Define the resampling rule
  resample_rule = {
      'd': 'D',
      'w': 'W',
      'm': 'M',
      'q': 'Q',
      'y': 'Y'
  }.get(frequency.lower())

  if not resample_rule:
      raise ValueError("Frequency must be one of 'd', 'w', 'm', 'q', 'y'.")

  # Resample and calculate OHLC
  ohlc = data.resample(resample_rule).agg(['first', 'max', 'min', 'last'])
  ohlc.columns = ['Open', 'High', 'Low', 'Close']

  return ohlc