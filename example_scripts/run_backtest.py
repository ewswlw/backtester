from strategy_framework import Strategy
from strategy_5_volatility_regime import VolatilityRegimeStrategy
import pandas as pd
import numpy as np

def load_data():
    """Load and prepare data for backtesting"""
    # Replace this with your actual data loading logic
    df = pd.DataFrame()  # Your data loading here
    return df

def run_backtest(strategies, config):
    """Run backtest for multiple strategies"""
    results = {}
    
    for strategy in strategies:
        strategy_name = strategy.__class__.__name__
        metrics = strategy.backtest()
        results[strategy_name] = metrics
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T
    
    # Format percentage columns
    pct_columns = ['total_return', 'annual_return', 'daily_vol', 'downside_vol', 
                  'max_drawdown', 'win_rate', 'market_coverage']
    
    for col in pct_columns:
        if col in results_df.columns:
            results_df[col] = results_df[col] * 100
    
    return results_df

if __name__ == "__main__":
    # Load data
    df = load_data()
    
    # Initialize strategies
    strategies = [
        VolatilityRegimeStrategy(df)
    ]
    
    # Run backtest
    results = run_backtest(strategies, None)
    print("\nBacktest Results:")
    print("================")
    print(results)
