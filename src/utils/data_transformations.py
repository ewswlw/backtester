import pandas as pd
import logging
from typing import Dict

def convert_er_ytd_to_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts year-to-date excess return data in all columns to custom indices starting at 100.
    Assumes that all columns contain year-to-date excess return data in percentage format.

    :param df: DataFrame containing year-to-date excess return data in percentage format.
    :return: DataFrame with custom indices columns calculated from daily returns.
    """

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
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
