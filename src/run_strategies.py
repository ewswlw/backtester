import os
import yaml
import pandas as pd
import vectorbt as vbt
from typing import Dict, Any

from strategies.ma_strategy import MovingAverageStrategy
from strategies.buy_and_hold_strategy import BuyAndHoldStrategy
from strategies.complex_regime_strategy import ComplexRegimeStrategy
from strategies.trend_risk_strategy import TrendRiskStrategy

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_path = os.path.join(project_root, 'config', 'backtest_config.yaml')
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(data_path: str) -> pd.DataFrame:
    """Load and prepare data for backtesting."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df.ffill().bfill()

def analyze_portfolio(portfolio: vbt.Portfolio, name: str) -> pd.Series:
    """Analyze portfolio performance and return key metrics."""
    # Calculate drawdown series
    drawdown = portfolio.drawdown()
    max_dd = abs(drawdown.min())  # Get absolute value of max drawdown
    
    # Calculate returns
    total_return = portfolio.total_return()
    
    # Calculate CAGR
    days = (portfolio.wrapper.index[-1] - portfolio.wrapper.index[0]).days
    cagr = ((1 + total_return) ** (365 / days) - 1) * 100
    
    # Calculate risk-adjusted metrics
    sharpe = portfolio.sharpe_ratio()
    sortino = portfolio.sortino_ratio()
    calmar = abs(total_return) / max_dd if max_dd > 0 else float('inf')
    
    # Calculate trade metrics
    stats = portfolio.stats()
    win_rate = stats['Win Rate [%]'] / 100 if 'Win Rate [%]' in stats else 0
    
    # Store metrics
    metrics = pd.Series({
        'Strategy': name,
        'Total Return (%)': total_return * 100,
        'CAGR (%)': cagr,
        'Max Drawdown (%)': max_dd * 100,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Calmar Ratio': calmar,
        'Win Rate (%)': win_rate * 100,
        'Total Trades': stats.get('Total Trades', 0)
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
    price_series = df['cad_ig_er_index']
    
    # Initialize strategies
    strategies = {
        'MA': MovingAverageStrategy(config),
        'Buy & Hold': BuyAndHoldStrategy(config),
        'Complex Regime': ComplexRegimeStrategy(config),
        'Trend Risk': TrendRiskStrategy(config)
    }
    
    # Run strategies and collect results
    results = {}
    metrics = []
    
    for name, strategy in strategies.items():
        try:
            print(f"\nRunning {name} strategy...")
            portfolio = strategy.backtest(price_series)
            results[name] = portfolio
            
            # Calculate and store metrics
            strategy_metrics = analyze_portfolio(portfolio, name)
            metrics.append(strategy_metrics)
            
            # Print detailed stats
            print(f"\n{name} Strategy Stats:")
            print(portfolio.stats())
            
        except Exception as e:
            print(f"Error running {name} strategy: {str(e)}")
            continue
    
    if metrics:
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(metrics)
        comparison_df.set_index('Strategy', inplace=True)
        
        # Print comparison
        print("\nStrategy Comparison:")
        print(comparison_df.round(2))
        
        # Find best strategy
        best_strategy = comparison_df['Total Return (%)'].idxmax()
        print(f"\nBest performing strategy: {best_strategy}")
        print(f"Total Return: {comparison_df.loc[best_strategy, 'Total Return (%)']:.2f}%")
        print(f"CAGR: {comparison_df.loc[best_strategy, 'CAGR (%)']:.2f}%")
        print(f"Max Drawdown: {comparison_df.loc[best_strategy, 'Max Drawdown (%)']:.2f}%")
        print(f"Sharpe Ratio: {comparison_df.loc[best_strategy, 'Sharpe Ratio']:.2f}")

if __name__ == "__main__":
    main()
