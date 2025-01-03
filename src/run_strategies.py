import os
import yaml
import pandas as pd
from typing import Dict, Any

from strategies.ma_strategy import MovingAverageStrategy
from strategies.buy_and_hold_strategy import BuyAndHoldStrategy

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    # Get the directory containing run_strategies.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to project root
    project_root = os.path.dirname(current_dir)
    # Construct path to config file
    config_path = os.path.join(project_root, 'config', 'backtest_config.yaml')
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(data_path: str) -> pd.DataFrame:
    """Load and prepare data for backtesting."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    df = pd.read_csv(data_path, dtype={'Date': str, 'cad_ig_er_index': float})
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df.ffill().bfill()

def format_stats(portfolio) -> pd.Series:
    """Format portfolio statistics for comparison."""
    stats = portfolio.stats()
    
    # Calculate CAGR
    total_years = len(portfolio.wrapper.index) / 252  # Assuming 252 trading days per year
    cagr = (portfolio.total_return() + 1) ** (1 / total_years) - 1 if total_years > 0 else 0
    
    return pd.Series({
        'Start': portfolio.wrapper.index[0],
        'End': portfolio.wrapper.index[-1],
        'Total Return [%]': portfolio.total_return() * 100,  # Convert to percentage
        'CAGR [%]': cagr * 100,  # Convert to percentage
        'Max Drawdown [%]': portfolio.max_drawdown() * 100,  # Convert to percentage
        'Sharpe Ratio': portfolio.sharpe_ratio(),
        'Sortino Ratio': portfolio.sortino_ratio(),
        'Calmar Ratio': portfolio.calmar_ratio(),
        'Omega Ratio': portfolio.omega_ratio(),
        'Win Rate [%]': stats['Win Rate [%]'],
        'Total Trades': stats['Total Trades'],
        'Avg Win/Loss': abs(stats['Avg Winning Trade [%]'] / stats['Avg Losing Trade [%]']) if stats['Avg Losing Trade [%]'] != 0 else float('inf')
    })

def main():
    # Load config and data
    config = load_config()
    
    # Get project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Construct data path
    data_path = os.path.join(project_root, config['data']['file_path'])
    print(f"\nLoading data from {data_path}")
    df = load_data(data_path)
    price_series = df[config['data']['target_column']]
    
    # Initialize strategies
    strategies = {
        'MA': MovingAverageStrategy(config),
        'Buy & Hold': BuyAndHoldStrategy(config)
    }
    
    # Run strategies and collect results
    results = {}
    for name, strategy in strategies.items():
        try:
            # Run backtest
            print(f"\nRunning {name} strategy...")
            portfolio = strategy.backtest(price_series)
            results[name] = portfolio
            
            # Print detailed stats
            print(f"\n{name} Strategy Stats:")
            print(portfolio.stats())
            
        except Exception as e:
            print(f"Error running {name} strategy: {str(e)}")
    
    # Print comparison summary
    if results:
        print("\nStrategy Comparison Summary:")
        comparison_df = pd.DataFrame({
            name: format_stats(portfolio)
            for name, portfolio in results.items()
        }).T
        
        # Sort strategies by total return
        comparison_df = comparison_df.sort_values('Total Return [%]', ascending=False)
        
        # Format floating point numbers
        float_cols = comparison_df.select_dtypes(include=['float64']).columns
        comparison_df[float_cols] = comparison_df[float_cols].round(4)
        
        print(comparison_df)
        
        # Print best performing strategy
        best_strategy = comparison_df.index[0]
        print(f"\nBest performing strategy: {best_strategy}")
        print(f"Total Return: {comparison_df.loc[best_strategy, 'Total Return [%]']:.2f}%")
        print(f"CAGR: {comparison_df.loc[best_strategy, 'CAGR [%]']:.2f}%")
        print(f"Sharpe Ratio: {comparison_df.loc[best_strategy, 'Sharpe Ratio']:.2f}")

if __name__ == "__main__":
    main()
