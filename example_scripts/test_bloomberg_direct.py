import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from xbbg import blp

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.core.bloomberg_fetcher import fetch_bloomberg_data

def test_bloomberg_fetch():
    """
    Test the fetch_bloomberg_data function with some common Bloomberg securities and fields.
    """
    try:
        # Define a test mapping of securities and fields
        mapping = {
            ('SPX Index', 'PX_LAST'): 'SPX_Price',
            ('SPX Index', 'VOLUME'): 'SPX_Volume',
            ('INDU Index', 'PX_LAST'): 'DJIA_Price',
            ('VIX Index', 'PX_LAST'): 'VIX_Level'
        }
        
        # Calculate dates
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        print(f"\nFetching data from {start_date} to {end_date}")
        print("Securities and fields:")
        for (ticker, field), col_name in mapping.items():
            print(f"  {ticker} - {field} -> {col_name}")
        
        # Fetch the data
        df = fetch_bloomberg_data(
            mapping=mapping,
            start_date=start_date,
            end_date=end_date,
            periodicity='D',
            align_start=True
        )
        
        # Display results
        print("\nData shape:", df.shape)
        print("\nFirst few rows:")
        print(df.head())
        
        print("\nData info:")
        print(df.info())
        
        print("\nBasic statistics:")
        print(df.describe())
        
        return True
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return False
    
if __name__ == "__main__":
    print("Testing Bloomberg Data Fetching")
    print("============================")
    
    success = test_bloomberg_fetch()
    
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!")
        sys.exit(1)
