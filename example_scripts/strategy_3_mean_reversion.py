from strategy_framework import Strategy
import pandas as pd
import numpy as np
from scipy import stats

class MeanReversionStrategy(Strategy):
    """
    Strategy 3: Mean Reversion Strategy
    
    Logic:
    - Uses z-scores to identify overbought/oversold conditions in spreads
    - Incorporates volatility-adjusted thresholds
    - Uses multiple timeframes for confirmation
    - Includes trend filter to avoid fighting strong trends
    """
    
    def __init__(self, df: pd.DataFrame, 
                 lookback_periods: list = [20, 60],
                 zscore_threshold: float = 1.5,  
                 vol_window: int = 20,
                 trend_window: int = 200):
        super().__init__(df)
        self.lookback_periods = lookback_periods
        self.zscore_threshold = zscore_threshold
        self.vol_window = vol_window
        self.trend_window = trend_window
    
    def _calculate_zscore(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling z-score"""
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        zscore = (series - rolling_mean) / rolling_std
        return zscore
    
    def _calculate_rolling_efficiency_ratio(self, series: pd.Series, window: int) -> pd.Series:
        """
        Calculate rolling efficiency ratio as a proxy for mean reversion tendency
        Values closer to 0 indicate more mean reversion, values closer to 1 indicate trending
        """
        # Calculate absolute price change over window
        abs_change = abs(series.diff(window))
        
        # Calculate sum of absolute price changes within window
        rolling_abs_changes = abs(series.diff()).rolling(window).sum()
        
        # Calculate efficiency ratio
        efficiency_ratio = abs_change / rolling_abs_changes
        
        return efficiency_ratio
    
    def generate_signals(self) -> pd.Series:
        """Generate trading signals based on mean reversion conditions"""
        signals = pd.Series(False, index=self.df.index)
        
        # Store intermediate signals for debugging
        signal_components = {}
        
        # Calculate signals for each lookback period
        entry_signals = []
        for period in self.lookback_periods:
            # Calculate z-scores for spreads
            cad_zscore = self._calculate_zscore(self.df['cad_oas'], period)
            us_hy_zscore = self._calculate_zscore(self.df['us_hy_oas'], period)
            
            # Mean reversion signals - long when CAD spreads are wide relative to US
            spread_signal = (cad_zscore > self.zscore_threshold) & \
                          (us_hy_zscore < -self.zscore_threshold/2)  
            
            entry_signals.append(spread_signal)
            signal_components[f'spread_signal_{period}'] = spread_signal
        
        # Combine signals from different timeframes
        combined_signal = pd.concat(entry_signals, axis=1).any(axis=1)
        signal_components['combined_spread_signals'] = combined_signal
        
        # Calculate efficiency ratio instead of Hurst (more reliable)
        efficiency_ratio = self._calculate_rolling_efficiency_ratio(self.df['cad_oas'], self.vol_window)
        mean_reverting = efficiency_ratio < 0.5  
        signal_components['mean_reverting'] = mean_reverting
        
        # Volatility filter
        vol = self.df['cad_oas'].rolling(self.vol_window).std()
        vol_percentile = vol.rolling(252).rank(pct=True)
        vol_filter = vol_percentile < 0.8  
        signal_components['vol_filter'] = vol_filter
        
        # Trend filter (don't fight strong trends)
        ma_fast = self.df[self.target_col].rolling(50).mean()
        ma_slow = self.df[self.target_col].rolling(200).mean()
        trend_filter = ma_fast > ma_slow
        signal_components['trend_filter'] = trend_filter
        
        # Combine all filters with more relaxed conditions
        signals = combined_signal & \
                 mean_reverting.fillna(False) & \
                 vol_filter.fillna(False) & \
                 trend_filter.fillna(False)
        
        # Print signal statistics
        print("\nMean Reversion Strategy Signal Analysis:")
        print("=======================================")
        total_days = len(signals)
        for name, component in signal_components.items():
            true_count = component.sum()
            pct = (true_count / total_days) * 100
            print(f"{name}: {true_count} days ({pct:.1f}% of time)")
        
        return signals.fillna(False)
