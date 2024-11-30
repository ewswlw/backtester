"""Data validation utilities for Bloomberg data pipeline."""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def find_duplicate_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Find and analyze duplicate rows in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        Tuple[pd.DataFrame, List[str]]: Duplicate rows and list of insights
    """
    duplicates = df[df.index.duplicated(keep=False)].sort_index()
    insights = []
    
    if not duplicates.empty:
        insights.append(f"Found {len(duplicates)} duplicate rows")
        # Group duplicates to analyze patterns
        dup_groups = duplicates.groupby(duplicates.index)
        for date, group in dup_groups:
            if not group.iloc[0].equals(group.iloc[1]):
                insights.append(f"Data mismatch for {date}:")
                for col in df.columns:
                    if not group[col].nunique() == 1:
                        insights.append(f"  - {col}: {group[col].tolist()}")
    
    return duplicates, insights

def handle_missing_values(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """
    Handle missing values in DataFrame using specified method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        method (str): Method to handle missing values ('ffill', 'bfill', 'interpolate')
    
    Returns:
        pd.DataFrame: DataFrame with handled missing values
    """
    if method == 'ffill':
        df_clean = df.fillna(method='ffill')
    elif method == 'bfill':
        df_clean = df.fillna(method='bfill')
    elif method == 'interpolate':
        df_clean = df.interpolate(method='time')
    else:
        raise ValueError(f"Unsupported missing value handling method: {method}")
    
    # Log missing value statistics
    missing_before = df.isna().sum()
    missing_after = df_clean.isna().sum()
    
    for col in df.columns:
        if missing_before[col] > 0:
            logger.info(f"Column {col}: Filled {missing_before[col]} missing values using {method}")
            if missing_after[col] > 0:
                logger.warning(f"Column {col}: {missing_after[col]} values still missing after {method}")
    
    return df_clean

def validate_dataframe(df: pd.DataFrame, name: str, 
                      expected_cols: Optional[List[str]] = None,
                      value_ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> List[str]:
    """
    Enhanced validation for DataFrame with detailed checks and insights.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        name (str): Name of the dataset for logging
        expected_cols (List[str], optional): Expected column names
        value_ranges (Dict[str, Tuple[float, float]], optional): Expected value ranges per column
    
    Returns:
        List[str]: List of validation insights and warnings
    """
    insights = []
    
    # Basic DataFrame info
    insights.append("\n" + "="*50)
    insights.append(f"Data Validation for: {name}")
    insights.append("="*50 + "\n")
    
    # Check index
    if not isinstance(df.index, pd.DatetimeIndex):
        insights.append("WARNING: Index is not DatetimeIndex")
    else:
        insights.append(f"Date Range: {df.index.min()} to {df.index.max()}")
        # Check for gaps in dates
        business_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
        missing_dates = business_days.difference(df.index)
        if len(missing_dates) > 0:
            insights.append(f"WARNING: Found {len(missing_dates)} missing business days")
            insights.append(f"First few missing dates: {list(missing_dates[:5])}")
    
    # Column validation
    if expected_cols:
        missing_cols = set(expected_cols) - set(df.columns)
        extra_cols = set(df.columns) - set(expected_cols)
        if missing_cols:
            insights.append(f"WARNING: Missing expected columns: {missing_cols}")
        if extra_cols:
            insights.append(f"WARNING: Found unexpected columns: {extra_cols}")
    
    # Data type checks
    insights.append("\nColumn Data Types:")
    for col in df.columns:
        insights.append(f"  {col}: {df[col].dtype}")
        if df[col].dtype == 'object':
            insights.append(f"WARNING: Column {col} contains object (string) data")
    
    # Missing values
    missing = df.isna().sum()
    if missing.any():
        insights.append("\nMissing Values:")
        for col, count in missing.items():
            if count > 0:
                insights.append(f"  {col}: {count} missing values")
    
    # Value range validation
    if value_ranges:
        insights.append("\nValue Range Validation:")
        for col, (expected_min, expected_max) in value_ranges.items():
            if col in df.columns:
                actual_min = df[col].min()
                actual_max = df[col].max()
                if actual_min < expected_min or actual_max > expected_max:
                    insights.append(f"WARNING: {col} values outside expected range:")
                    insights.append(f"  Expected: [{expected_min}, {expected_max}]")
                    insights.append(f"  Actual: [{actual_min}, {actual_max}]")
    
    # Duplicate check
    duplicates, dup_insights = find_duplicate_rows(df)
    if dup_insights:
        insights.append("\nDuplicate Analysis:")
        insights.extend(dup_insights)
    
    # Basic statistics
    insights.append("\nBasic Statistics:")
    stats = df.describe()
    for col in df.columns:
        insights.append(f"\n{col}:")
        insights.append(f"  Mean: {stats[col]['mean']:.2f}")
        insights.append(f"  Std: {stats[col]['std']:.2f}")
        insights.append(f"  Min: {stats[col]['min']:.2f}")
        insights.append(f"  Max: {stats[col]['max']:.2f}")
    
    return insights
