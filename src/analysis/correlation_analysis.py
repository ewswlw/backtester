import pandas as pd
import numpy as np
from typing import Dict, Tuple
import os

def load_data() -> pd.DataFrame:
    """Load and prepare data for analysis."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_path = os.path.join(project_root, 'pulling_data', 'backtest_data.csv')
    
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate returns for all numeric columns."""
    returns_df = df.select_dtypes(include=[np.number]).pct_change()
    returns_df.columns = [f"{col}_ret" for col in returns_df.columns]
    return returns_df

def calculate_correlations(df: pd.DataFrame, target_col: str, 
                         lookbacks: list = [1, 3, 6, 12]) -> Dict[str, pd.DataFrame]:
    """Calculate correlations at different lookback periods."""
    
    # Get returns
    returns = calculate_returns(df)
    target_returns = returns[f"{target_col}_ret"]
    
    results = {}
    
    # Current level correlations
    level_corr = df.corrwith(df[target_col])
    results['levels'] = pd.DataFrame({
        'correlation': level_corr,
        'abs_correlation': abs(level_corr)
    }).sort_values('abs_correlation', ascending=False)
    
    # Return correlations at different lookbacks
    for months in lookbacks:
        # Forward returns of target
        fwd_returns = target_returns.shift(-months)
        
        # Correlations with current indicators
        level_fwd_corr = df.corrwith(fwd_returns)
        ret_fwd_corr = returns.corrwith(fwd_returns)
        
        # Combine and sort
        combined_corr = pd.concat([level_fwd_corr, ret_fwd_corr])
        results[f'{months}m_fwd'] = pd.DataFrame({
            'correlation': combined_corr,
            'abs_correlation': abs(combined_corr)
        }).sort_values('abs_correlation', ascending=False)
    
    return results

def analyze_predictive_power(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, Dict]:
    """Analyze the predictive power of indicators."""
    
    # Calculate z-scores for all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    zscore_df = df[numeric_cols].apply(lambda x: (x - x.rolling(252).mean()) / x.rolling(252).std())
    
    # Forward returns of target at different horizons
    returns = {}
    for months in [1, 3, 6, 12]:
        returns[f'{months}m_fwd_ret'] = df[target_col].pct_change(months).shift(-months)
    
    # Combine returns
    returns_df = pd.DataFrame(returns)
    
    # Calculate information coefficients (rank correlation)
    ic_results = {}
    for col in zscore_df.columns:
        if col != target_col:
            ic_results[col] = {
                'ic_mean': {},
                'ic_std': {},
                'ic_ir': {}  # Information ratio
            }
            for ret_col in returns_df.columns:
                # Rolling 12-month rank correlation
                rolling_ic = zscore_df[col].rolling(252).corr(returns_df[ret_col], method='spearman')
                
                ic_results[col]['ic_mean'][ret_col] = rolling_ic.mean()
                ic_results[col]['ic_std'][ret_col] = rolling_ic.std()
                ic_results[col]['ic_ir'][ret_col] = rolling_ic.mean() / rolling_ic.std()
    
    # Convert to DataFrame
    ic_df = pd.DataFrame({
        col: {
            f"{period}_ic_mean": results['ic_mean'][period]
            for period in returns_df.columns
        } for col, results in ic_results.items()
    }).T
    
    return ic_df, ic_results

def main():
    """Run correlation analysis."""
    # Load data
    df = load_data()
    target_col = 'cad_ig_er_index'
    
    print("\nCalculating correlations...")
    corr_results = calculate_correlations(df, target_col)
    
    print("\nLevel Correlations:")
    print(corr_results['levels'].head(10))
    
    print("\nForward 3-month Correlations:")
    print(corr_results['3m_fwd'].head(10))
    
    print("\nAnalyzing predictive power...")
    ic_df, ic_results = analyze_predictive_power(df, target_col)
    
    print("\nTop predictive indicators (3-month horizon):")
    print(ic_df.sort_values('3m_fwd_ret_ic_mean', ascending=False).head(10))
    
    # Save results
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    results_dir = os.path.join(project_root, 'analysis_results')
    os.makedirs(results_dir, exist_ok=True)
    
    ic_df.to_csv(os.path.join(results_dir, 'predictive_power.csv'))
    for period, corr_df in corr_results.items():
        corr_df.to_csv(os.path.join(results_dir, f'correlations_{period}.csv'))
    
    print(f"\nResults saved to {results_dir}")

if __name__ == "__main__":
    main()
