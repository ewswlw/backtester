import os
import sys
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any
from pathlib import Path

# Add project root to path for interactive window
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Try both import styles to support both script and interactive modes
try:
    from src.strategies.ma_strategy import MovingAverageStrategy
    from src.strategies.buy_and_hold_strategy import BuyAndHoldStrategy
    from src.strategies.hy_timing_strategy import HYTimingStrategy
except ImportError:
    from strategies.ma_strategy import MovingAverageStrategy
    from strategies.buy_and_hold_strategy import BuyAndHoldStrategy
    from strategies.hy_timing_strategy import HYTimingStrategy

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent / 'config' / 'backtest_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert relative path to absolute path
    if 'data' in config and 'file_path' in config['data']:
        config['data']['file_path'] = str(Path(__file__).parent.parent / config['data']['file_path'])
    
    return config

def load_data(data_path: str) -> pd.DataFrame:
    """Load and prepare data for backtesting."""
    print(f"\nLoading data from: {data_path}")
    
    # Load data
    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded CSV with shape: {df.shape}")
        
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Set Date as index
        df.set_index('Date', inplace=True)
        print("Set Date as index")
        
        # Sort by date
        df.sort_index(inplace=True)
        print("Sorted by date")
        
        # Handle duplicates - keep last observation for each month
        df = df[~df.index.duplicated(keep='last')]
        print(f"Removed duplicates, new shape: {df.shape}")
        
        # Forward fill missing values
        df.fillna(method='ffill', inplace=True)
        # Backward fill any remaining missing values at the start
        df.fillna(method='bfill', inplace=True)
        print("Filled missing values")
        
        # Print column names and data types
        print("\nColumns and dtypes:")
        for col in df.columns:
            print(f"{col}: {df[col].dtype}")
        
        return df
        
    except Exception as e:
        print(f"\nError loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def analyze_portfolio(portfolio, name: str) -> pd.Series:
    """Analyze portfolio performance and return key metrics."""
    metrics = {}
    metrics['Strategy'] = name
    
    # Get portfolio value series and returns
    if hasattr(portfolio, 'value'):
        value_series = portfolio.value()
        returns = portfolio.returns()
        
        # Calculate basic metrics
        total_return = ((value_series.iloc[-1] / value_series.iloc[0]) - 1) * 100
        metrics['Total Return [%]'] = total_return
        
        # Calculate years
        years = (value_series.index[-1] - value_series.index[0]).days / 365.25
        ann_return = ((1 + total_return/100) ** (1/years) - 1) * 100
        metrics['Annualized Return [%]'] = ann_return
        
        # Calculate volatility (adjust for monthly data)
        monthly_vol = returns.std() * np.sqrt(12)
        ann_vol = monthly_vol * 100
        metrics['Annualized Volatility [%]'] = ann_vol
        metrics['Sharpe Ratio'] = ann_return / ann_vol if ann_vol != 0 else 0
        
        # Calculate drawdown metrics
        rolling_max = value_series.expanding().max()
        drawdown = ((value_series - rolling_max) / rolling_max) * 100
        metrics['Max Drawdown [%]'] = drawdown.min()
        metrics['Calmar Ratio'] = abs(ann_return / metrics['Max Drawdown [%]']) if metrics['Max Drawdown [%]'] != 0 else float('inf')
        
        # Monthly metrics
        monthly_returns = returns * 100
        metrics['Monthly Win Rate [%]'] = (monthly_returns > 0).mean() * 100
        metrics['Monthly Best Return [%]'] = monthly_returns.max()
        metrics['Monthly Worst Return [%]'] = monthly_returns.min()
        metrics['Avg Monthly Return [%]'] = monthly_returns.mean()
        metrics['Monthly Volatility [%]'] = monthly_returns.std() * 100
        
    else:
        # For non-portfolio series (e.g., benchmark), calculate basic metrics
        value_series = portfolio
        returns = value_series.pct_change().fillna(0)
        
        # Calculate basic metrics
        total_return = ((value_series.iloc[-1] / value_series.iloc[0]) - 1) * 100
        metrics['Total Return [%]'] = total_return
        
        # Calculate years
        years = (value_series.index[-1] - value_series.index[0]).days / 365.25
        ann_return = ((1 + total_return/100) ** (1/years) - 1) * 100
        metrics['Annualized Return [%]'] = ann_return
        
        # Calculate volatility (adjust for monthly data)
        monthly_vol = returns.std() * np.sqrt(12)
        ann_vol = monthly_vol * 100
        metrics['Annualized Volatility [%]'] = ann_vol
        metrics['Sharpe Ratio'] = ann_return / ann_vol if ann_vol != 0 else 0
        
        # Calculate drawdown metrics
        rolling_max = value_series.expanding().max()
        drawdown = ((value_series - rolling_max) / rolling_max) * 100
        metrics['Max Drawdown [%]'] = drawdown.min()
        metrics['Calmar Ratio'] = abs(ann_return / metrics['Max Drawdown [%]']) if metrics['Max Drawdown [%]'] != 0 else float('inf')
        
        # Monthly metrics
        monthly_returns = returns * 100
        metrics['Monthly Win Rate [%]'] = (monthly_returns > 0).mean() * 100
        metrics['Monthly Best Return [%]'] = monthly_returns.max()
        metrics['Monthly Worst Return [%]'] = monthly_returns.min()
        metrics['Avg Monthly Return [%]'] = monthly_returns.mean()
        metrics['Monthly Volatility [%]'] = monthly_returns.std() * 100
    
    return pd.Series(metrics)

def main():
    """Main function to run all strategies."""
    try:
        # Load configuration
        config = load_config()
        
        # Load data
        data_path = Path(__file__).parent.parent / 'pulling_data' / 'backtest_data.csv'
        if not data_path.exists():
            print(f"\nError: Data file not found at {data_path}")
            print("Available files in pulling_data directory:")
            pulling_data_dir = data_path.parent
            for file in pulling_data_dir.glob('*'):
                print(f"- {file.name}")
            return
        
        print(f"\nLoading data from: {data_path}")
        df = load_data(str(data_path))
        
        if df is None or df.empty:
            print("Error: Failed to load data or data is empty")
            return
        
        # Print data info
        print("\nData Summary:")
        print("-" * 80)
        print(f"Shape: {df.shape}")
        print(f"Date Range: {df.index[0]} to {df.index[-1]}")
        print(f"Columns: {', '.join(df.columns)}")
        
        # Get price series for simple strategies
        price_series = df[config['data']['target_column']]
        print(f"\nUsing target column: {config['data']['target_column']}")
        
        # Initialize strategies
        strategies = {
            'MA': MovingAverageStrategy(config),
            'HY Timing': HYTimingStrategy(config),
            'Buy & Hold': BuyAndHoldStrategy(config)
        }
        
        # Run strategies and collect results
        results = []
        print("\nRunning strategies...")
        
        # First run buy & hold as benchmark
        print("\nRunning Buy & Hold (Benchmark)...")
        benchmark = strategies['Buy & Hold'].backtest(price_series)
        benchmark_stats = analyze_portfolio(benchmark, 'Buy & Hold (Benchmark)')
        results.append(benchmark_stats)
        
        # Run other strategies
        for name, strategy in strategies.items():
            if name == 'Buy & Hold':
                continue
                
            print(f"\nRunning {name} strategy...")
            try:
                portfolio = strategy.backtest(price_series)
                stats = analyze_portfolio(portfolio, name)
                results.append(stats)
                
                # Print detailed portfolio stats
                print(f"\n{name} Strategy Performance vs Benchmark:")
                print("=" * 80)
                print("\nKey Performance Metrics:")
                print(f"{'Metric':<25} {'Strategy':>12} {'Benchmark':>12} {'Diff':>12}")
                print("-" * 80)
                
                metrics_to_show = [
                    ('Total Return', 'Total Return [%]', '%'),
                    ('Ann. Return', 'Annualized Return [%]', '%'),
                    ('Sharpe Ratio', 'Sharpe Ratio', ''),
                    ('Max Drawdown', 'Max Drawdown [%]', '%')
                ]
                
                for label, metric, suffix in metrics_to_show:
                    strategy_val = stats[metric]
                    benchmark_val = benchmark_stats[metric]
                    diff = strategy_val - benchmark_val
                    print(f"{label:<25} {strategy_val:>11.2f}{suffix} {benchmark_val:>11.2f}{suffix} {diff:>11.2f}{suffix}")
                
                print("\nMonthly Statistics:")
                print("-" * 80)
                monthly_metrics = [
                    ('Win Rate', 'Monthly Win Rate [%]', '%'),
                    ('Avg Return', 'Avg Monthly Return [%]', '%'),
                    ('Monthly Vol', 'Monthly Volatility [%]', '%'),
                    ('Best Month', 'Monthly Best Return [%]', '%'),
                    ('Worst Month', 'Monthly Worst Return [%]', '%')
                ]
                
                for label, metric, suffix in monthly_metrics:
                    val = stats[metric]
                    print(f"{label:<25} {val:>11.2f}{suffix}")
            except Exception as e:
                print(f"\nError running {name} strategy: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        comparison_df.set_index('Strategy', inplace=True)
        
        # Sort metrics by importance
        metric_order = [
            'Total Return [%]', 'Annualized Return [%]', 'Annualized Volatility [%]',
            'Sharpe Ratio', 'Calmar Ratio', 'Max Drawdown [%]',
            'Monthly Win Rate [%]', 'Avg Monthly Return [%]', 'Monthly Volatility [%]',
            'Monthly Best Return [%]', 'Monthly Worst Return [%]'
        ]
        comparison_df = comparison_df[metric_order]
        
        # Format the DataFrame for display
        pd.set_option('display.float_format', lambda x: '{:,.2f}'.format(x))
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('display.colheader_justify', 'right')
        
        print("\nStrategy Comparison:")
        print("=" * 120)
        
        # Rename columns for better readability
        display_df = comparison_df.copy()
        display_df.columns = [
            'Total Return', 'Ann. Return', 'Ann. Vol',
            'Sharpe', 'Calmar', 'Max DD',
            'Win Rate', 'Avg Return', 'Monthly Vol',
            'Best Month', 'Worst Month'
        ]
        
        # Add % symbol to percentage columns
        pct_columns = ['Total Return', 'Ann. Return', 'Ann. Vol', 'Max DD',
                      'Win Rate', 'Avg Return', 'Monthly Vol', 'Best Month', 'Worst Month']
        for col in pct_columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}%")
        
        # Format ratio columns
        ratio_columns = ['Sharpe', 'Calmar']
        for col in ratio_columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}")
        
        # Print with better formatting
        print("\nFull Performance Comparison:")
        print("-" * 120)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(display_df.to_string())
        print("-" * 120)
        
        # Find best performing strategy
        best_strategy = comparison_df.loc[comparison_df['Total Return [%]'].idxmax()]
        print(f"\nBest Performing Strategy: {best_strategy.name}")
        print(f"Total Return: {best_strategy['Total Return [%]']:.2f}%")
        print(f"Annualized Return: {best_strategy['Annualized Return [%]']:.2f}%")
        print(f"Sharpe Ratio: {best_strategy['Sharpe Ratio']:.2f}")
        
    except Exception as e:
        print(f"\nError running strategies: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
