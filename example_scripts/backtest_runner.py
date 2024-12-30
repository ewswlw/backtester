import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import os

# Filter warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)  # For div by zero, etc
warnings.filterwarnings('ignore', category=FutureWarning)   # For pandas operations
warnings.filterwarnings('ignore', category=UserWarning)     # For matplotlib
np.seterr(all='ignore')  # Ignore numpy warnings
pd.options.mode.chained_assignment = None  # Ignore pandas chained assignment warnings

from strategy_1_momentum import DualMomentumStrategy
from strategy_2_regime import MacroRegimeStrategy
from strategy_3_mean_reversion import MeanReversionStrategy
from strategy_4_multi_factor import MultiFactorStrategy
from strategy_5_volatility_regime import VolatilityRegimeStrategy
from strategy_6_adaptive_trend import AdaptiveTrendStrategy
# from strategy_7_ml_ensemble import MLEnsembleStrategy
from strategy_8_combined import CombinedStrategy
from strategy_analysis import print_strategy_analysis

def load_data() -> pd.DataFrame:
    """Load and preprocess data"""
    data_path = Path(__file__).parent.parent / 'raw_data' / 'df.csv'
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    return df

def print_data_overview(df: pd.DataFrame):
    """Print data overview"""
    print("\nData Overview:")
    print("=============")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Number of rows: {len(df)}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nLast few rows:")
    print(df.tail())
    print("\nBasic statistics:")
    print(df.describe())

def print_metrics_table(results_df: pd.DataFrame):
    """Print a formatted table of strategy metrics"""
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    # Group metrics by category
    metric_groups = {
        'Returns': ['total_return', 'annual_return', 'daily_vol', 'downside_vol', 'max_drawdown'],
        'Risk-Adjusted': ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio'],
        'Trading': ['total_trades', 'win_rate', 'avg_win', 'avg_loss', 'profit_factor', 'market_coverage', 'avg_hold_time'],
        'Regime': ['high_vol_return', 'low_vol_return', 'avg_vix', 'avg_regime'],
        'Risk': ['return_skew', 'return_kurtosis', 'return_autocorr', 'time_in_drawdown', 'recovery_ratio'],
        'Streaks': ['max_consecutive_wins', 'max_consecutive_losses']
    }
    
    print("\nStrategy Performance Analysis")
    print("============================\n")
    
    for group_name, metrics in metric_groups.items():
        print(f"\n{group_name} Metrics:")
        print("-" * (len(group_name) + 8))
        
        # Select metrics for this group
        group_df = results_df[metrics].copy()
        
        # Format percentages
        pct_metrics = ['total_return', 'annual_return', 'daily_vol', 'downside_vol', 
                      'max_drawdown', 'win_rate', 'market_coverage', 'high_vol_return', 
                      'low_vol_return', 'time_in_drawdown']
        
        # Format each column
        for col in group_df.columns:
            if col in pct_metrics:
                group_df[col] = group_df[col].map('{:.1%}'.format)
            else:
                group_df[col] = group_df[col].map('{:.2f}'.format)
        
        print(group_df.to_string())
        print("\n" + "=" * 80)

def main():
    # Load and prepare data
    df = load_data()
    print_data_overview(df)
    
    # Create strategy instances
    strategies = [
        DualMomentumStrategy(df),
        MacroRegimeStrategy(df),
        MeanReversionStrategy(df),
        MultiFactorStrategy(df),
        VolatilityRegimeStrategy(df),
        AdaptiveTrendStrategy(df),
        # MLEnsembleStrategy(df),
        CombinedStrategy(df)
    ]
    
    # Run backtests and collect results
    results = {}
    for strategy in strategies:
        strategy_name = strategy.__class__.__name__
        metrics = strategy.backtest()
        results[strategy_name] = metrics
    
    # Convert to DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index')
    
    # Print detailed metrics
    print_metrics_table(results_df)
    
    # Print strategy analysis
    print_strategy_analysis(df, strategies)
    
    # Save results
    results_path = Path(__file__).parent.parent / 'results' / 'backtest_results.csv'
    results_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(results_path)
    print(f"\nResults saved to: {os.path.abspath(results_path)}")

if __name__ == "__main__":
    main()
