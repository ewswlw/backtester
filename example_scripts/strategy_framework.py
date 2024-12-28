from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from scipy import stats
import vectorbt as vbt
from typing import Dict, Tuple, List

class Strategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'cad_ig_er_ytd_index'):
        self.df = df.copy()
        self.target_col = target_col
        self.name = self.__class__.__name__
        
    @abstractmethod
    def generate_signals(self) -> pd.Series:
        """Generate trading signals"""
        pass
        
    def _calculate_returns(self, signals: pd.Series) -> pd.Series:
        """Calculate strategy returns"""
        price = self.df[self.target_col]
        returns = price.pct_change()
        strategy_returns = returns * signals.shift(1).fillna(0)
        return strategy_returns
        
    def _calculate_metrics(self, returns: pd.Series, signals: pd.Series) -> dict:
        """Calculate comprehensive strategy metrics"""
        # Basic returns metrics
        total_return = (1 + returns).prod() - 1
        n_years = (self.df.index[-1] - self.df.index[0]).days / 365.25
        annual_return = (1 + total_return) ** (1/n_years) - 1
        
        # Risk metrics
        daily_vol = returns.std() * np.sqrt(252)
        downside_vol = returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0
        
        # Calculate drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdowns = cumulative / running_max - 1
        max_drawdown = drawdowns.min()
        
        # Risk-adjusted returns
        sharpe = annual_return / daily_vol if daily_vol != 0 else 0
        sortino = annual_return / downside_vol if downside_vol != 0 else 0
        calmar = -annual_return / max_drawdown if max_drawdown != 0 else 0
        
        # Trading metrics
        n_trades = (signals != signals.shift(1)).sum()
        win_rate = (returns[returns != 0] > 0).mean() if len(returns[returns != 0]) > 0 else 0
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
        
        # Calculate profit factor safely
        total_gains = returns[returns > 0].sum()
        total_losses = abs(returns[returns < 0].sum())
        profit_factor = total_gains / total_losses if total_losses != 0 else np.inf
        
        # Market exposure metrics
        market_coverage = signals.mean()
        avg_hold_time = len(signals) / max(n_trades, 1)
        
        # Risk regime metrics
        high_vol = self.df['vix'] > self.df['vix'].rolling(252).mean()
        returns_high_vol = returns[high_vol]
        returns_low_vol = returns[~high_vol]
        
        # Statistical metrics
        returns_no_zero = returns[returns != 0].astype(float)
        returns_no_zero = returns_no_zero[~np.isnan(returns_no_zero) & ~np.isinf(returns_no_zero)]
        
        if len(returns_no_zero) > 2:  # Need at least 3 points for skew/kurtosis
            try:
                skew = float(stats.skew(returns_no_zero))
                kurtosis = float(stats.kurtosis(returns_no_zero))
            except:
                skew = 0
                kurtosis = 0
        else:
            skew = 0
            kurtosis = 0
        
        # Autocorrelation
        autocorr = returns.autocorr() if len(returns) > 1 else 0
        
        # Calculate time in drawdown
        drawdown_periods = (cumulative < running_max).sum()
        time_in_drawdown = drawdown_periods / len(returns)
        
        # Calculate recovery ratio
        max_dd_duration = self._calculate_max_drawdown_duration(cumulative, running_max)
        recovery_ratio = total_return / max_dd_duration if max_dd_duration > 0 else np.inf
        
        return {
            # Return metrics
            'total_return': total_return,
            'annual_return': annual_return,
            'daily_vol': daily_vol,
            'downside_vol': downside_vol,
            'max_drawdown': max_drawdown,
            
            # Risk-adjusted metrics
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            
            # Trading metrics
            'total_trades': n_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'market_coverage': market_coverage,
            'avg_hold_time': avg_hold_time,
            
            # Regime performance
            'high_vol_return': returns_high_vol.mean() * 252 if len(returns_high_vol) > 0 else 0,
            'low_vol_return': returns_low_vol.mean() * 252 if len(returns_low_vol) > 0 else 0,
            'avg_vix': self.df.loc[signals.astype(bool), 'vix'].mean(),
            'avg_regime': self.df.loc[signals.astype(bool), 'us_economic_regime'].mean(),
            
            # Statistical properties
            'return_skew': skew,
            'return_kurtosis': kurtosis,
            'return_autocorr': autocorr,
            'time_in_drawdown': time_in_drawdown,
            'recovery_ratio': recovery_ratio,
            
            # Streaks
            'max_consecutive_wins': self._max_consecutive(returns > 0),
            'max_consecutive_losses': self._max_consecutive(returns < 0)
        }
    
    def _max_consecutive(self, series: pd.Series) -> int:
        """Calculate maximum consecutive True values"""
        if not any(series):
            return 0
        groups = (series != series.shift()).cumsum()
        counts = series.groupby(groups).cumsum()
        return int(counts.max()) if len(counts) > 0 else 0
    
    def _calculate_max_drawdown_duration(self, cumulative: pd.Series, running_max: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        in_drawdown = cumulative < running_max
        if not any(in_drawdown):
            return 0
            
        # Find drawdown periods
        drawdown_groups = (in_drawdown != in_drawdown.shift()).cumsum()
        drawdown_lengths = in_drawdown.groupby(drawdown_groups).sum()
        
        return int(drawdown_lengths.max()) if len(drawdown_lengths) > 0 else 0
    
    def backtest(self) -> dict:
        """Run backtest and return performance metrics"""
        signals = self.generate_signals()
        returns = self._calculate_returns(signals)
        metrics = self._calculate_metrics(returns, signals)
        return metrics

def calculate_benchmark_metrics(df: pd.DataFrame, target_col: str = 'cad_ig_er_ytd_index') -> Dict:
    """
    Calculate buy & hold benchmark metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the target column
    target_col : str
        Name of the target column
        
    Returns:
    --------
    Dict
        Dictionary containing benchmark metrics
    """
    # Calculate returns manually first for verification
    price_series = df[target_col]
    total_return = (price_series.iloc[-1] / price_series.iloc[0]) - 1
    n_years = (df.index[-1] - df.index[0]).days / 365.25
    annual_return = (1 + total_return) ** (1/n_years) - 1
    
    print("\nBuy & Hold Manual Calculation:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annual Return: {annual_return:.2%}")
    
    # Run vectorbt backtest
    portfolio = vbt.Portfolio.from_holding(close=df[target_col], freq='1D')
    
    metrics = {
        'strategy_name': 'Buy & Hold',
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': portfolio.sharpe_ratio(),
        'max_drawdown': portfolio.max_drawdown(),
        'total_trades': 1  # Buy & hold is just one trade
    }
    
    return metrics
