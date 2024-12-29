"""Test vectorbt metrics to ensure they work correctly"""
import pandas as pd
import numpy as np
import vectorbt as vbt
from .strategy_framework import Strategy
from .strategy_0_buy_and_hold import BuyAndHoldStrategy

def test_portfolio_metrics(df: pd.DataFrame, signals: pd.Series, price_col: str = 'cad_ig_er_ytd_index'):
    """Test each vectorbt metric individually"""
    print("Testing vectorbt metrics...")
    
    # Create test portfolio
    try:
        portfolio = vbt.Portfolio.from_signals(
            close=df[price_col],
            entries=signals,
            exits=~signals,
            freq='1D'
        )
        print("[PASS] Portfolio creation successful")
    except Exception as e:
        print(f"[FAIL] Portfolio creation failed: {str(e)}")
        return
    
    # Test returns calculation
    try:
        returns = portfolio.returns()
        print(f"[PASS] Returns calculation successful")
        print(f"  First few returns: {returns.head()}")
    except Exception as e:
        print(f"[FAIL] Returns calculation failed: {str(e)}")
    
    # Test total return
    try:
        total_return = portfolio.total_return()
        print(f"[PASS] Total return calculation successful: {total_return:.2%}")
    except Exception as e:
        print(f"[FAIL] Total return calculation failed: {str(e)}")
    
    # Test drawdown
    try:
        drawdown = portfolio.drawdown()
        max_dd = drawdown.min()  # Most negative value
        print(f"[PASS] Drawdown calculation successful")
        print(f"  Max drawdown: {max_dd:.2%}")
    except Exception as e:
        print(f"[FAIL] Drawdown calculation failed: {str(e)}")
        print("Trying alternative drawdown calculation...")
        try:
            returns = portfolio.returns()
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdowns = cum_returns / running_max - 1
            max_dd = drawdowns.min()
            print(f"[PASS] Alternative drawdown calculation successful: {max_dd:.2%}")
        except Exception as e2:
            print(f"[FAIL] Alternative drawdown calculation failed: {str(e2)}")
    
    # Test Sharpe ratio
    try:
        sharpe = portfolio.sharpe_ratio()
        print(f"[PASS] Sharpe ratio calculation successful: {sharpe:.2f}")
    except Exception as e:
        print(f"[FAIL] Sharpe ratio calculation failed: {str(e)}")
        try:
            # Try alternative method
            returns = portfolio.returns()
            excess_returns = returns - 0.02/252  # Assuming 2% risk-free rate
            sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
            print(f"[PASS] Alternative Sharpe ratio calculation successful: {sharpe:.2f}")
        except Exception as e2:
            print(f"[FAIL] Alternative Sharpe ratio calculation failed: {str(e2)}")
    
    # Test trades
    try:
        trades = portfolio.trades
        n_trades = len(trades)
        print(f"[PASS] Trades calculation successful: {n_trades} trades")
    except Exception as e:
        print(f"[FAIL] Trades calculation failed: {str(e)}")
        try:
            # Try alternative method
            signals_diff = signals.diff()
            n_trades = (signals_diff != 0).sum()
            print(f"[PASS] Alternative trades calculation successful: {n_trades} trades")
        except Exception as e2:
            print(f"[FAIL] Alternative trades calculation failed: {str(e2)}")
    
    return portfolio

def test_vectorbt_metrics(df: pd.DataFrame, strategy: Strategy) -> dict:
    """Test vectorbt metrics calculation for a strategy"""
    
    # Get buy and hold benchmark
    benchmark = BuyAndHoldStrategy(df)
    benchmark_signals = benchmark.generate_signals()
    benchmark_metrics = benchmark.backtest()
    
    # Get strategy signals and metrics
    strategy_signals = strategy.generate_signals()
    strategy_metrics = strategy.backtest()
    
    # Calculate relative metrics
    relative_metrics = {
        'excess_return': strategy_metrics['annual_return'] - benchmark_metrics['annual_return'],
        'relative_sharpe': strategy_metrics['sharpe_ratio'] - benchmark_metrics['sharpe_ratio'],
        'drawdown_improvement': benchmark_metrics['max_drawdown'] - strategy_metrics['max_drawdown']
    }
    
    print(f"\nTesting {strategy.__class__.__name__} vs Buy & Hold:")
    print("=" * 50)
    print(f"Strategy Annual Return: {strategy_metrics['annual_return']:.2%}")
    print(f"Buy & Hold Annual Return: {benchmark_metrics['annual_return']:.2%}")
    print(f"Excess Return: {relative_metrics['excess_return']:.2%}")
    print(f"Strategy Sharpe: {strategy_metrics['sharpe_ratio']:.2f}")
    print(f"Buy & Hold Sharpe: {benchmark_metrics['sharpe_ratio']:.2f}")
    print(f"Relative Sharpe: {relative_metrics['relative_sharpe']:.2f}")
    print(f"Drawdown Improvement: {relative_metrics['drawdown_improvement']:.2%}")
    
    return relative_metrics

if __name__ == "__main__":
    print("\nCreating test dataset...")
    # Small test dataset
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    prices = pd.Series(np.random.random(len(dates)).cumsum(), index=dates)
    signals = pd.Series(np.random.choice([True, False], len(dates)), index=dates)
    df = pd.DataFrame({'cad_ig_er_ytd_index': prices})
    
    print("\nRunning tests...")
    # Run test
    test_portfolio_metrics(df, signals)
