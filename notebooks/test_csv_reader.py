import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.csv_exporter import read_csv_to_df

def test_all_combinations(file_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Test all combinations of fill and start_date_align parameters.
    
    Args:
        file_path (Path): Path to the CSV file to test
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of test results
    """
    fill_options = [None, 'ffill', 'bfill', 'interpolate']
    align_options = ['no', 'yes']
    results = {}
    
    print(f"\nTesting file: {file_path.name}")
    print("=" * 80)
    
    for fill in fill_options:
        for align in align_options:
            combo_name = f"fill={fill}, align={align}"
            print(f"\nTesting combination: {combo_name}")
            print("-" * 80)
            
            df = read_csv_to_df(file_path, fill=fill, start_date_align=align)
            results[combo_name] = df
            
            # Print DataFrame info
            print("\nDataFrame Info:")
            df.info()
            
            # Print first and last 10 rows
            print("\nFirst 10 rows:")
            print(df.head(10))
            print("\nLast 10 rows:")
            print(df.tail(10))
            
            # Print basic stats
            print("\nBasic Statistics:")
            print(f"Shape: {df.shape}")
            print(f"Date Range: {df.index.min()} to {df.index.max()}")
            print(f"Total NaN values: {df.isna().sum().sum()}")
            print(f"Number of columns with all NaN: {(df.isna().sum() == len(df)).sum()}")
            
            print("\n" + "=" * 80)
    
    return results

def main():
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # List of CSV files to test with correct paths
    csv_files = [
        project_root / "data/sprds_data.csv",
        project_root / "data/derv_data.csv",
        project_root / "data/er_ytd_data.csv"
    ]
    
    # Test each file
    all_results = {}
    for file_path in csv_files:
        if file_path.exists():
            results = test_all_combinations(file_path)
            all_results[file_path.name] = results
        else:
            print(f"File not found: {file_path}")
    
    return all_results

if __name__ == "__main__":
    all_results = main()
