import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import os
import json
from IPython.display import display

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
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    # Format dates and durations
    results_df['start_date'] = pd.to_datetime(results_df['start_date']).dt.strftime('%m/%d/%Y')
    results_df['end_date'] = pd.to_datetime(results_df['end_date']).dt.strftime('%m/%d/%Y')
    results_df['period'] = ((pd.to_datetime(results_df['end_date']) - pd.to_datetime(results_df['start_date'])).dt.days).astype(str) + ' days'
    
    # Format percentages
    for col in ['total_return', 'annual_return', 'daily_vol', 'downside_vol', 'max_drawdown', 'win_rate', 'market_coverage', 'exposure']:
        results_df[col] = results_df[col].multiply(100)
    
    # Format durations
    for col in ['avg_win_duration', 'avg_loss_duration']:
        results_df[col] = results_df[col].apply(lambda x: f"{x.days} days" if isinstance(x, pd.Timedelta) else "NaT")
    
    # Rename columns for display
    display_names = {
        'start_date': 'Start',
        'end_date': 'End',
        'period': 'Period',
        'total_return': 'Total Return (%)',
        'annual_return': 'Annual Return (%)',
        'daily_vol': 'Annualized Volatility (%)',
        'downside_vol': 'Downside Volatility (%)',
        'max_drawdown': 'Max Drawdown (%)',
        'sharpe_ratio': 'Sharpe Ratio',
        'sortino_ratio': 'Sortino Ratio',
        'calmar_ratio': 'Calmar Ratio',
        'total_trades': 'Total Trades',
        'win_rate': 'Win Rate (%)',
        'avg_win': 'Avg Win',
        'avg_loss': 'Avg Loss',
        'profit_factor': 'Profit Factor',
        'avg_win_duration': 'Avg Winning Trade Duration',
        'avg_loss_duration': 'Avg Losing Trade Duration',
        'market_coverage': 'Market Coverage (%)',
        'exposure': 'Exposure (%)',
        'value_at_risk': 'Value at Risk',
        'cvar': 'Conditional VaR'
    }
    
    results_df = results_df.rename(columns=display_names)
    
    # Order metrics in groups
    metric_groups = {
        'Overview': ['Start', 'End', 'Period'],
        'Returns': ['Total Return (%)', 'Annual Return (%)', 'Annualized Volatility (%)', 
                   'Downside Volatility (%)', 'Max Drawdown (%)'],
        'Risk Metrics': ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Value at Risk', 'Conditional VaR'],
        'Trading Stats': ['Total Trades', 'Win Rate (%)', 'Avg Win', 'Avg Loss', 'Profit Factor',
                         'Avg Winning Trade Duration', 'Avg Losing Trade Duration'],
        'Exposure': ['Market Coverage (%)', 'Exposure (%)']
    }
    
    # Print each group
    print("\nStrategy Performance Metrics:")
    print("=" * 80)
    
    for group_name, metrics in metric_groups.items():
        print(f"\n{group_name}:")
        print("-" * 80)
        group_df = results_df[metrics]
        
        # Apply styling
        styled_df = group_df.style.format(precision=2, thousands=",")
        styled_df = styled_df.set_properties(**{'text-align': 'center'})
        styled_df = styled_df.set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center')]}
        ])
        
        display(styled_df)

def main():
    # Load and prepare data
    df = load_data()
    print_data_overview(df)
    
    # Backtest configuration
    backtest_config = {
        'start_date': '2002-10-31',  # Match vectorbt notebook start date
        'end_date': '2024-12-27',    # Match vectorbt notebook end date
        'rebalance_freq': '1D',      # Daily rebalancing
        'init_cash': 100.0,          # Start with $100
        'fees': 0.001,               # 0.1% trading fee
        'slippage': 0.001,           # 0.1% slippage
        'risk_free': 0.02            # 2% risk-free rate
    }
    
    # Create strategy instances
    strategies = [
        AdaptiveTrendStrategy(df),
        CombinedStrategy(df),
        MultiFactorStrategy(df),
        MacroRegimeStrategy(df),
        VolatilityRegimeStrategy(df),
        MeanReversionStrategy(df),
        DualMomentumStrategy(df)
    ]
    
    # Run backtests and collect results
    results = {}
    for strategy in strategies:
        strategy_name = strategy.__class__.__name__
        metrics = strategy.backtest(**backtest_config)
        results[strategy_name] = metrics
    
    # Convert to DataFrame and sort columns to match vectorbt notebook
    results_df = pd.DataFrame.from_dict(results, orient='index')
    
    # Print detailed metrics
    print_metrics_table(results_df)
    
    # Print strategy analysis
    print_strategy_analysis(df, strategies)
    
    # Save results with timestamp
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    results_path = Path(__file__).parent.parent / 'results' / f'backtest_results_{timestamp}.csv'
    results_path.parent.mkdir(exist_ok=True)
    
    # Save results and configuration
    results_df.to_csv(results_path)
    config_path = results_path.with_name(f'backtest_config_{timestamp}.json')
    with open(config_path, 'w') as f:
        json.dump(backtest_config, f, indent=4, default=str)
    
    print(f"\nResults saved to: {os.path.abspath(results_path)}")
    print(f"Configuration saved to: {os.path.abspath(config_path)}")
    
if __name__ == "__main__":
    main()
