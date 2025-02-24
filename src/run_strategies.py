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
    from src.strategies.hy_timing_strategy import HYTimingStrategy
except ImportError:
    from strategies.ma_strategy import MovingAverageStrategy
    from strategies.buy_and_hold_strategy import BuyAndHoldStrategy
    from strategies.hy_timing_strategy import HYTimingStrategy

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

def analyze_portfolio(portfolio, name: str) -> pd.Series:
    """Analyze portfolio performance and return key metrics."""
    # Convert Portfolio object to Series if needed
    if hasattr(portfolio, 'returns'):
        portfolio_series = pd.Series(portfolio.value(), index=portfolio.wrapper.index)
    else:
        portfolio_series = portfolio
        
    # Calculate returns
    returns = portfolio_series.pct_change().fillna(0)
    
    # Calculate basic metrics
    total_return = ((portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1) * 100
    
    # Calculate years based on actual date difference
    start_date = portfolio_series.index[0]
    end_date = portfolio_series.index[-1]
    years = (end_date - start_date).days / 365.25
    
    # Calculate annualized return properly
    ann_return = ((1 + total_return/100) ** (1/years) - 1) * 100
    
    # Calculate volatility (adjust for monthly data)
    monthly_vol = returns.std() * np.sqrt(12)  # Annualize monthly volatility
    ann_vol = monthly_vol * 100
    
    # Calculate Sharpe Ratio (assuming 0% risk-free rate for simplicity)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    
    # Calculate Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(12) * 100  # Annualize monthly downside volatility
    sortino = ann_return / downside_vol if downside_vol != 0 else 0
    
    # Calculate drawdown
    rolling_max = portfolio_series.expanding().max()
    drawdown = ((portfolio_series - rolling_max) / rolling_max) * 100
    max_drawdown = drawdown.min()
    
    # Calculate Calmar Ratio
    calmar = abs(ann_return / max_drawdown) if max_drawdown != 0 else 0
    
    # Calculate Omega Ratio
    threshold = 0
    gains = returns[returns > threshold]
    losses = returns[returns <= threshold]
    omega = abs(gains.mean() / losses.mean()) if len(losses) > 0 and losses.mean() != 0 else float('inf')
    
    # Calculate Value at Risk and Conditional VaR
    var_95 = np.percentile(returns, 5) * 100
    cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
    
    # Trading statistics
    position_changes = (returns != 0).astype(int).diff()
    total_trades = (position_changes != 0).sum() // 2
    
    metrics = pd.Series({
        'Total Return [%]': total_return,
        'Annualized Return [%]': ann_return,
        'Annualized Volatility [%]': ann_vol,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Calmar Ratio': calmar,
        'Omega Ratio': omega,
        'Max Drawdown [%]': max_drawdown,
        'Value at Risk [%]': var_95,
        'Conditional Value at Risk [%]': cvar_95,
        'Strategy': name,
        'Data Frequency': f"{(portfolio_series.index[1] - portfolio_series.index[0]).days} days",
        'Start': portfolio_series.index[0],
        'End': portfolio_series.index[-1],
        'Period': portfolio_series.index[-1] - portfolio_series.index[0],
        'Start Value': portfolio_series.iloc[0],
        'End Value': portfolio_series.iloc[-1],
        'Benchmark Return [%]': total_return,
        'Max Gross Exposure [%]': 100.0,
        'Total Fees Paid': 0.0,
        'Max Drawdown Duration': f"{(-drawdown > 0).groupby((-drawdown > 0).ne(-drawdown > 0).cumsum()).cumcount().max()} days",
        'Total Trades': total_trades,
        'Total Closed Trades': total_trades - 1 if total_trades > 0 else 0,
        'Total Open Trades': 1 if total_trades > 0 else 0,
        'Open Trade PnL': portfolio_series.iloc[-1] - portfolio_series.iloc[0] if total_trades > 0 else 0,
        'Win Rate [%]': np.nan,
        'Best Trade [%]': np.nan,
        'Worst Trade [%]': np.nan,
        'Avg Winning Trade [%]': np.nan,
        'Avg Losing Trade [%]': np.nan,
        'Avg Winning Trade Duration': pd.NaT,
        'Avg Losing Trade Duration': pd.NaT,
        'Profit Factor': np.nan,
        'Expectancy': np.nan
    })
    
    return metrics

def main():
    """Main function to run all strategies."""
    # Load configuration
    config = load_config()
    
    # Load data
    data_path = config['data']['file_path']
    print(f"\nLoading data from {data_path}")
    df = load_data(data_path)
    
    # Get price series for simple strategies
    price_series = df[config['data']['target_column']]
    
    # Initialize strategies
    strategies = {
        'MA': MovingAverageStrategy(config),
        'HY Timing': HYTimingStrategy(config),
        'Buy & Hold': BuyAndHoldStrategy(config)
    }
    
    # Optimize MA strategy parameters
    print("\nOptimizing MA strategy parameters...")
    ma_strategy = strategies['MA']
    optimal_params = ma_strategy.optimize_parameters(price_series)
    print(f"\nOptimal parameters for MA strategy:")
    print(f"MA Window: {optimal_params['ma_window']}")
    print(f"Expected Sharpe: {optimal_params['sharpe']:.2f}")
    print(f"Expected Total Return: {optimal_params['total_return']:.2%}")
    
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
