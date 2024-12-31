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
        
        # For buy and hold (constant True signals), look at all returns
        # For active strategies, only look at returns when invested
        is_buy_and_hold = signals.all() and len(signals.unique()) == 1
        if is_buy_and_hold:
            win_rate = (returns > 0).mean()
        else:
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
    
    def backtest(self, start_date: str = None, end_date: str = None, 
                rebalance_freq: str = '1D', init_cash: float = 100.0, 
                fees: float = 0.0, slippage: float = 0.0,
                risk_free: float = 0.0) -> dict:
        """Run backtest and return performance metrics
        
        Args:
            start_date (str, optional): Start date for backtest. Format: 'YYYY-MM-DD'
            end_date (str, optional): End date for backtest. Format: 'YYYY-MM-DD'
            rebalance_freq (str, optional): Rebalancing frequency. Default: '1D'
            init_cash (float, optional): Initial cash amount. Default: 100.0
            fees (float, optional): Trading fees as percentage. Default: 0.0
            slippage (float, optional): Slippage as percentage. Default: 0.0
            risk_free (float, optional): Risk-free rate for Sharpe ratio. Default: 0.0
            
        Returns:
            dict: Dictionary containing backtest metrics
        """
        # Generate signals and filter date range
        signals = self.generate_signals()
        if start_date:
            signals = signals[start_date:]
        if end_date:
            signals = signals[:end_date]
            
        # Get price data for the same period
        price = self.df[self.target_col][signals.index[0]:signals.index[-1]]
        
        # Create portfolio with vectorbt
        portfolio = vbt.Portfolio.from_signals(
            price,
            signals,
            init_cash=init_cash,
            fees=fees,
            slippage=slippage,
            freq=rebalance_freq,
            log=True  # Enable logging for detailed metrics
        )
        
        # Get portfolio metrics
        total_return = portfolio.total_return()
        returns = portfolio.returns()
        returns_stats = returns.vbt.returns(freq='1D', year_freq='365D')
        
        # Core metrics using returns_stats
        annual_return = returns_stats.annualized()
        daily_vol = returns_stats.annualized_volatility()
        
        # Calculate downside volatility manually using negative returns
        negative_returns = returns[returns < 0]
        downside_vol = np.sqrt(252) * np.sqrt(np.mean(negative_returns**2)) if len(negative_returns) > 0 else 0
        
        max_drawdown = returns_stats.max_drawdown()
        
        # Risk-adjusted metrics using returns_stats
        sharpe = returns_stats.sharpe_ratio()
        sortino = annual_return / downside_vol if downside_vol != 0 else 0
        calmar = -annual_return / max_drawdown if max_drawdown != 0 else 0
        
        # Trading metrics
        trades = portfolio.trades
        n_trades = len(trades)
        win_rate = trades.win_rate() if n_trades > 0 else 0
        
        # Calculate average win/loss using trades accessor
        avg_win = trades.winning.returns.mean() if len(trades.winning) > 0 else 0
        avg_loss = trades.losing.returns.mean() if len(trades.losing) > 0 else 0
        profit_factor = trades.profit_factor() if n_trades > 0 else np.inf
        
        # Duration metrics
        avg_win_duration = trades.winning.duration.mean() if len(trades.winning) > 0 else pd.NaT
        avg_loss_duration = trades.losing.duration.mean() if len(trades.losing) > 0 else pd.NaT
        
        # Market coverage and exposure
        market_coverage = signals.mean()
        exposure = portfolio.net_exposure()
        
        # Risk metrics using returns_stats
        var = returns_stats.value_at_risk(cutoff=0.05)  # 5% VaR
        cvar = returns_stats.cond_value_at_risk(cutoff=0.05)  # 5% CVaR
        
        # Return metrics dictionary
        return {
            'start_date': signals.index[0],
            'end_date': signals.index[-1],
            'total_return': total_return,
            'annual_return': annual_return,
            'daily_vol': daily_vol,
            'downside_vol': downside_vol,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'total_trades': n_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_win_duration': avg_win_duration,
            'avg_loss_duration': avg_loss_duration,
            'market_coverage': market_coverage,
            'exposure': exposure,
            'value_at_risk': var,
            'cvar': cvar
        }
