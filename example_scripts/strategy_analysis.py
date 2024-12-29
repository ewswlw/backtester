import pandas as pd
import numpy as np
from typing import List, Dict
import vectorbt as vbt
from .strategy_framework import Strategy
from .strategy_0_buy_and_hold import BuyAndHoldStrategy

def calculate_annual_return(total_return: float, days: int) -> float:
    """Calculate annualized return from total return and number of days"""
    years = days / 365.25
    return (1 + total_return) ** (1/years) - 1

def calculate_win_rate(returns: pd.Series) -> float:
    """Calculate win rate from returns series"""
    if len(returns) == 0:
        return 0
    return (returns > 0).mean()

def analyze_strategy_correlations(strategies: List[Strategy]) -> pd.DataFrame:
    """Analyze correlations between strategy signals"""
    signals = pd.DataFrame()
    
    # Get signals from each strategy
    for strategy in strategies:
        signals[strategy.__class__.__name__] = strategy.generate_signals()
    
    return signals.corr()

def analyze_market_conditions(df: pd.DataFrame, signals: pd.DataFrame, strategies: List[Strategy]) -> Dict:
    """Analyze when each strategy performs best using vectorbt"""
    conditions = {}
    
    # Get benchmark metrics using BuyAndHoldStrategy
    benchmark = next(s for s in strategies if isinstance(s, BuyAndHoldStrategy))
    benchmark_metrics = benchmark.backtest()
    
    for col in signals.columns:
        strategy_signals = signals[col]
        
        # Get metrics from strategy's own backtest method for consistency
        strategy = next(s for s in strategies if s.__class__.__name__ == col)
        metrics = strategy.backtest()
        
        conditions[col] = {
            # Core metrics (from strategy's backtest)
            'total_return': metrics['total_return'],
            'annual_return': metrics['annual_return'],
            'win_rate': metrics['win_rate'],
            'max_drawdown': metrics['max_drawdown'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'sortino_ratio': metrics['sortino_ratio'],
            'calmar_ratio': metrics['calmar_ratio'],
            'trades': metrics.get('trades', 1),  # Default to 1 for buy & hold
            
            # Additional risk metrics
            'volatility': metrics['daily_vol'],
            'avg_return': metrics.get('avg_return', metrics['annual_return']),  # Fallback to annual return
            'avg_win': metrics.get('avg_win', 0),
            'avg_loss': metrics.get('avg_loss', 0),
            
            # Comparison metrics vs benchmark
            'return_vs_bh': metrics['annual_return'] - benchmark_metrics['annual_return'],
            'win_rate_vs_bh': metrics['win_rate'] - benchmark_metrics['win_rate'],
            'drawdown_improvement': benchmark_metrics['max_drawdown'] - metrics['max_drawdown'],
            'sharpe_ratio_vs_bh': metrics['sharpe_ratio'] - benchmark_metrics['sharpe_ratio'],
            
            # Non-vectorbt metrics (market conditions)
            'signal_coverage': metrics.get('market_coverage', 1.0),  # Default to 1.0 for buy & hold
            'avg_vix': df['vix'][strategy_signals].mean(),
            'avg_regime': df['us_economic_regime'][strategy_signals].mean()
        }
    
    return conditions

def print_strategy_analysis(df: pd.DataFrame, strategies: List[Strategy]) -> None:
    """Print comprehensive strategy analysis"""
    # Ensure BuyAndHoldStrategy is first in the list for benchmark comparison
    if not isinstance(strategies[0], BuyAndHoldStrategy):
        strategies.insert(0, BuyAndHoldStrategy(df))
    
    # Get signals from each strategy
    signals = pd.DataFrame()
    for strategy in strategies:
        strategy_name = strategy.__class__.__name__
        signals[strategy_name] = strategy.generate_signals()
    
    # Analyze correlations
    correlations = analyze_strategy_correlations(strategies)
    
    # Analyze market conditions
    conditions = analyze_market_conditions(df, signals, strategies)
    
    # Print analysis for each strategy
    for strategy in strategies:
        strategy_name = strategy.__class__.__name__
        print(f"\n{strategy_name} Analysis:")
        print("=" * (len(strategy_name) + 9))
        
        # Get strategy metrics
        metrics = conditions[strategy_name]
        
        # Print core metrics
        print(f"Performance Metrics:")
        print(f"Total Return: {metrics['total_return']:.1%}")
        print(f"Annual Return: {metrics['annual_return']:.1%}")
        print(f"Win Rate: {metrics['win_rate']:.1%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.1%}")
        
        # Only print comparison metrics for non-benchmark strategies
        if not isinstance(strategy, BuyAndHoldStrategy):
            print(f"\nComparison vs Buy & Hold:")
            print(f"Excess Return: {metrics['return_vs_bh']:.1%}")
            print(f"Win Rate Difference: {metrics['win_rate_vs_bh']:.1%}")
            print(f"Drawdown Improvement: {metrics['drawdown_improvement']:.1%}")
            print(f"Relative Sharpe: {metrics['sharpe_ratio_vs_bh']:.2f}")
    
    # Print correlation matrix
    print("\nStrategy Signal Correlations:")
    print("============================")
    print(correlations.round(3))
