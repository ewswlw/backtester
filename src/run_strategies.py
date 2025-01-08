import os
import sys
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any

# Add project root to path for interactive window
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Try both import styles to support both script and interactive modes
try:
    from src.strategies.ma_strategy import MovingAverageStrategy
    from src.strategies.buy_and_hold_strategy import BuyAndHoldStrategy
except ImportError:
    from strategies.ma_strategy import MovingAverageStrategy
    from strategies.buy_and_hold_strategy import BuyAndHoldStrategy

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_path = os.path.join(project_root, 'config', 'backtest_config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert relative path to absolute path
    if 'data' in config and 'file_path' in config['data']:
        config['data']['file_path'] = os.path.join(project_root, config['data']['file_path'])
    
    return config

def load_data(data_path: str) -> pd.DataFrame:
    """Load and prepare data for backtesting."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df.ffill().bfill()

def analyze_portfolio(portfolio: pd.Series, name: str) -> pd.Series:
    """Analyze portfolio performance and return key metrics."""
    # Get returns with proper frequency settings
    returns = portfolio.returns()
    rets = returns.vbt.returns(freq='D')
    
    # Calculate annualized return manually
    total_return = rets.total()
    days = (portfolio.wrapper.index[-1] - portfolio.wrapper.index[0]).days
    ann_return = ((1 + total_return) ** (365.25 / days) - 1) * 100
    
    # Get other metrics from returns accessor
    metrics = {
        'Total Return [%]': total_return * 100,
        'Annualized Return [%]': ann_return,  # Use manually calculated annualized return
        'Annualized Volatility [%]': rets.annualized_volatility() * 100,
        'Sharpe Ratio': rets.sharpe_ratio(),
        'Sortino Ratio': rets.sortino_ratio(),
        'Calmar Ratio': rets.calmar_ratio(),
        'Omega Ratio': rets.omega_ratio(),
        'Max Drawdown [%]': rets.max_drawdown() * 100,
        'Value at Risk [%]': rets.value_at_risk() * 100,
        'Conditional Value at Risk [%]': rets.cond_value_at_risk() * 100,
        'Strategy': name
    }
    
    # Add trade statistics
    trade_stats = portfolio.stats()
    metrics.update({k: v for k, v in trade_stats.items() if k not in metrics})
    
    return pd.Series(metrics)

def main():
    """Main function to run all strategies."""
    # Load configuration
    config = load_config()
    
    # Load data
    data_path = config['data']['file_path']
    print(f"\nLoading data from {data_path}")
    df = load_data(data_path)
    
    # Get price series for simple strategies
    price_series = df['cad_ig_er_index']
    
    # Initialize strategies
    strategies = {
        'MA': MovingAverageStrategy(config),
        'Buy & Hold': BuyAndHoldStrategy(config)
    }
    
    # Run strategies and collect results
    results = []
    for name, strategy in strategies.items():
        print(f"\nRunning {name} strategy...")
        portfolio = strategy.backtest(price_series)
        
        # Get strategy stats
        stats = analyze_portfolio(portfolio, name)
        print(f"\n{name} Strategy Stats:")
        print(stats)
        results.append(stats)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    comparison_df.set_index('Strategy', inplace=True)
    
    print("\nStrategy Comparison:")
    print(comparison_df)
    
    # Find best performing strategy
    best_strategy = comparison_df.loc[comparison_df['Total Return [%]'].idxmax()]
    print(f"\nBest performing strategy: {best_strategy.name}")
    print(f"Total Return: {best_strategy['Total Return [%]']:.2f}%")
    print(f"Annualized Return: {best_strategy['Annualized Return [%]']:.2f}%")
    print(f"Annualized Volatility [%]: {best_strategy['Annualized Volatility [%]']:.2f}%")
    print(f"Sharpe Ratio: {best_strategy['Sharpe Ratio']:.2f}")

if __name__ == "__main__":
    main()
