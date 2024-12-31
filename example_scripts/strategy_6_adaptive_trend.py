from strategy_framework import Strategy
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.linear_model import LinearRegression

class AdaptiveTrendStrategy(Strategy):
    """
    Strategy 6: Adaptive Trend & Mean Reversion Strategy
    
    Logic:
    1. Decomposes price series into trend and cycle components
    2. Adapts between trend-following and mean-reversion based on:
       - Trend strength
       - Cycle amplitude
       - Market efficiency ratio
    3. Uses adaptive lookback periods based on market conditions
    4. Incorporates multiple timeframe confirmation
    """
    
    def __init__(self, df: pd.DataFrame,
                 cycle_lookbacks: list = [10, 20, 40],
                 efficiency_window: int = 10,
                 min_trend_strength: float = 0.4):
        super().__init__(df)
        self.cycle_lookbacks = cycle_lookbacks
        self.efficiency_window = efficiency_window
        self.min_trend_strength = min_trend_strength
        
    def _decompose_series(self, series: pd.Series, window: int) -> tuple:
        """Decompose series into trend and cycle using Butterworth filter"""
        # Normalize time series
        normalized = (series - series.mean()) / series.std()
        
        # Design Butterworth filter
        nyq = 0.5 * 1  # Normalized Nyquist frequency
        cutoff = 1 / window  # Cutoff frequency based on window
        order = 2  # Filter order
        
        # Create and apply filter
        b, a = signal.butter(order, cutoff/nyq, btype='low')
        trend = pd.Series(signal.filtfilt(b, a, normalized), index=series.index)
        cycle = normalized - trend
        
        return trend, cycle
        
    def _calculate_trend_strength(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate trend strength using R-squared of linear regression"""
        trend_strength = pd.Series(index=series.index)
        
        for i in range(window, len(series)):
            # Get window of data
            y = series.iloc[i-window:i]
            X = np.arange(window).reshape(-1, 1)
            
            # Fit linear regression
            reg = LinearRegression().fit(X, y)
            trend_strength.iloc[i] = reg.score(X, y)
            
        return trend_strength.fillna(0)
        
    def _calculate_cycle_score(self, cycle: pd.Series) -> pd.Series:
        """Calculate mean reversion opportunity in cycle component"""
        # Z-score of cycle
        cycle_zscore = (cycle - cycle.rolling(252).mean()) / cycle.rolling(252).std()
        
        # Score is high when cycle is extended
        cycle_score = -cycle_zscore  # Negative because high z-score = overbought
        
        return cycle_score
        
    def _calculate_adaptive_lookback(self) -> pd.Series:
        """Calculate adaptive lookback period based on market conditions"""
        # Use volatility to adjust lookback
        vol = self.df[self.target_col].pct_change().rolling(20).std() * np.sqrt(252)
        vol_ratio = vol / vol.rolling(252).mean()
        
        # Create series of lookbacks for each point in time
        base_lookback = np.mean(self.cycle_lookbacks)
        lookbacks = pd.Series(base_lookback, index=self.df.index)
        
        # Adjust lookbacks based on volatility
        adjusted_lookbacks = lookbacks * vol_ratio.fillna(1)
        
        # Clip lookbacks to reasonable range
        return adjusted_lookbacks.clip(min(self.cycle_lookbacks), max(self.cycle_lookbacks))
        
    def _calculate_market_efficiency_ratio(self) -> pd.Series:
        """Calculate market efficiency ratio"""
        price = self.df[self.target_col]
        
        # Directional movement over window
        dir_move = abs(price - price.shift(self.efficiency_window))
        
        # Total movement (path length)
        total_move = pd.Series(0, index=price.index)
        for i in range(1, self.efficiency_window + 1):
            total_move += abs(price - price.shift(i))
            
        # Calculate ratio
        efficiency_ratio = dir_move / total_move
        
        return efficiency_ratio
        
    def generate_signals(self) -> pd.Series:
        """Generate trading signals using adaptive approach"""
        signals = pd.Series(False, index=self.df.index)
        
        # Get adaptive lookback
        lookbacks = self._calculate_adaptive_lookback()
        avg_lookback = int(lookbacks.mean())
        
        # Decompose price series
        trend, cycle = self._decompose_series(self.df[self.target_col], avg_lookback)
        
        # Calculate component scores
        trend_strength = self._calculate_trend_strength(self.df[self.target_col], avg_lookback)
        cycle_score = self._calculate_cycle_score(cycle)
        efficiency_ratio = self._calculate_market_efficiency_ratio()
        
        # Print analysis
        print("\nAdaptive Trend Strategy Analysis:")
        print("================================")
        print(f"Average Trend Strength: {trend_strength.mean():.2f}")
        print(f"Average Efficiency Ratio: {efficiency_ratio.mean():.2f}")
        print(f"Average Lookback Period: {avg_lookback} days")
        
        # Determine regime with more lenient conditions
        trending_market = trend_strength > 0.3
        efficient_market = efficiency_ratio > 0.25
        
        # Generate signals based on regime with adjusted thresholds
        trend_signals = trending_market & (trend.diff() > 0)  
        reversion_signals = (~trending_market) & (cycle_score > 0.4)  
        
        # Combine signals - binary only (in/out)
        signals = trend_signals | reversion_signals
        
        # Resample to monthly frequency to match original implementation
        signals = signals.resample('M').last()
        
        # Forward fill the signals to match daily price data
        return signals.reindex(self.df.index).fillna(method='ffill').fillna(False)
