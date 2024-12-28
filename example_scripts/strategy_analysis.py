import pandas as pd
import numpy as np
from typing import List, Dict
from strategy_framework import Strategy

def analyze_strategy_correlations(strategies: List[Strategy]) -> pd.DataFrame:
    """Analyze correlations between strategy signals"""
    signals = pd.DataFrame()
    
    # Get signals from each strategy
    for strategy in strategies:
        signals[strategy.__class__.__name__] = strategy.generate_signals()
    
    return signals.corr()

def analyze_market_conditions(df: pd.DataFrame, signals: pd.DataFrame) -> Dict:
    """Analyze when each strategy performs best"""
    conditions = {}
    
    for col in signals.columns:
        strategy_signals = signals[col]
        
        # Calculate returns during strategy signals
        returns = df['cad_ig_er_ytd_index'].pct_change()
        strategy_returns = returns[strategy_signals]
        
        # Analyze market conditions during signals
        conditions[col] = {
            'avg_vix': df['vix'][strategy_signals].mean(),
            'avg_regime': df['us_economic_regime'][strategy_signals].mean(),
            'avg_return': strategy_returns.mean() * 252,  # Annualized
            'win_rate': (strategy_returns > 0).mean(),
            'max_drawdown': (strategy_returns + 1).cumprod().div(
                (strategy_returns + 1).cumprod().cummax()
            ).min(),
            'signal_coverage': strategy_signals.mean()
        }
    
    return conditions

def print_strategy_analysis(df: pd.DataFrame, strategies: List[Strategy]):
    """Print comprehensive strategy analysis"""
    # Get signals
    signals = pd.DataFrame()
    for strategy in strategies:
        signals[strategy.__class__.__name__] = strategy.generate_signals()
    
    # Calculate correlations
    corr = analyze_strategy_correlations(strategies)
    print("\nStrategy Signal Correlations:")
    print("============================")
    print(corr.round(2))
    
    # Analyze market conditions
    conditions = analyze_market_conditions(df, signals)
    print("\nStrategy Market Conditions Analysis:")
    print("==================================")
    for strategy, metrics in conditions.items():
        print(f"\n{strategy}:")
        print(f"Average VIX: {metrics['avg_vix']:.1f}")
        print(f"Average Economic Regime: {metrics['avg_regime']:.2f}")
        print(f"Average Annual Return: {metrics['avg_return']*100:.1f}%")
        print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"Maximum Drawdown: {metrics['max_drawdown']*100:.1f}%")
        print(f"Market Coverage: {metrics['signal_coverage']*100:.1f}%")
    
    # Find complementary strategies
    print("\nMost Complementary Strategy Pairs:")
    print("================================")
    min_corr = 1
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            correlation = corr.iloc[i,j]
            if correlation < min_corr:
                min_corr = correlation
                best_pair = (corr.columns[i], corr.columns[j])
    
    print(f"Best Pair: {best_pair[0]} and {best_pair[1]} (correlation: {min_corr:.2f})")
