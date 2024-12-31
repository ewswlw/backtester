from strategy_framework import Strategy
import pandas as pd
import numpy as np

class DualMomentumStrategy(Strategy):
    """
    Strategy 1: Dual Momentum Strategy
    
    Logic:
    - Combines absolute and relative momentum
    - Uses multiple lookback periods for robustness
    - Incorporates volatility regime filtering
    """
    
    def __init__(self, df: pd.DataFrame, 
                 lookback_periods: list = [20, 60, 120],
                 vol_window: int = 20,
                 vol_threshold: float = 1.5):
        super().__init__(df)
        self.lookback_periods = lookback_periods
        self.vol_window = vol_window
        self.vol_threshold = vol_threshold
        
    def generate_signals(self) -> pd.Series:
        signals = pd.Series(False, index=self.df.index)
        
        # Calculate returns for each lookback period
        momentum_signals = []
        for period in self.lookback_periods:
            # Absolute momentum (trend following)
            abs_momentum = self.df[self.target_col].pct_change(period) > 0
            
            # Relative momentum (compared to other assets)
            other_assets = [col for col in self.df.columns if col != self.target_col 
                          and 'er_ytd_index' in col]
            
            rel_returns = []
            for asset in other_assets:
                rel_momentum = (self.df[self.target_col].pct_change(period) >
                              self.df[asset].pct_change(period))
                rel_returns.append(rel_momentum)
            
            rel_momentum_signal = pd.concat(rel_returns, axis=1).all(axis=1)
            
            # Combine absolute and relative momentum
            momentum_signals.append(abs_momentum & rel_momentum_signal)
        
        # Combine signals from different lookback periods
        combined_momentum = pd.concat(momentum_signals, axis=1).mean(axis=1) > 0.5
        
        # Volatility regime filter
        rolling_vol = self.df[self.target_col].pct_change().rolling(self.vol_window).std()
        vol_regime = rolling_vol < (rolling_vol.rolling(252).mean() * self.vol_threshold)
        
        # Final signals
        signals = combined_momentum & vol_regime
        
        return signals.fillna(False)
