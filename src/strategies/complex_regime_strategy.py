import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

from .strategy_base import StrategyBase, BacktestConfig

@dataclass
class ComplexRegimeConfig:
    """Configuration for Complex Regime strategy."""
    min_holding_period: int = 15  # Minimum holding period in days
    signal_threshold: float = 0.2  # Threshold for regime score

class ComplexRegimeStrategy(StrategyBase):
    """
    Complex regime strategy combining price-based signals with regime detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the strategy with configuration."""
        super().__init__(BacktestConfig(**config['backtest_settings']))
        self.regime_config = ComplexRegimeConfig(**config['strategies']['ComplexRegime'])
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on market regime detection."""
        if isinstance(data, pd.Series):
            price_series = data
        else:
            price_series = data.iloc[:, 0]  # Use first column as price series
            
        # Calculate returns and volatility
        returns = price_series.pct_change()
        vol = returns.rolling(window=21).std() * np.sqrt(252)
        
        # Calculate trend indicators
        ma_fast = price_series.rolling(window=21).mean()  # 1-month MA
        ma_med = price_series.rolling(window=63).mean()   # 3-month MA
        ma_slow = price_series.rolling(window=126).mean() # 6-month MA
        
        # Calculate momentum indicators
        mom_1m = returns.rolling(window=21).mean() * 252  # 1-month momentum
        mom_3m = returns.rolling(window=63).mean() * 252  # 3-month momentum
        
        # Calculate regime scores
        def zscore(x: pd.Series, window: int = 252) -> pd.Series:
            return (x - x.rolling(window=window).mean()) / x.rolling(window=window).std()
        
        # Trend regime score (multiple timeframes)
        trend_score = (
            0.4 * (ma_fast > ma_med).astype(float) +
            0.3 * (ma_med > ma_slow).astype(float) +
            0.3 * zscore(price_series, window=126)
        )
        
        # Momentum regime score
        mom_score = (
            0.6 * zscore(mom_1m, window=126) +
            0.4 * zscore(mom_3m, window=126)
        )
        
        # Volatility regime score (inverse relationship)
        vol_score = -zscore(vol, window=126)
        
        # Combine regime scores
        regime_score = (
            0.4 * trend_score +
            0.4 * mom_score +
            0.2 * vol_score
        )
        
        # Generate signals
        signals = pd.Series(False, index=price_series.index)
        
        # Long signal conditions:
        # 1. Positive regime score (bullish conditions)
        # 2. Volatility not too high
        # 3. Positive momentum
        signals[
            (regime_score > self.regime_config.signal_threshold) &  # Bullish regime
            (vol < 0.15) &         # Moderate volatility
            (mom_1m > 0)           # Positive momentum
        ] = True
        
        # Apply minimum holding period
        min_hold = self.regime_config.min_holding_period
        for i in range(min_hold, len(signals)):
            if signals.iloc[i-min_hold:i].any():
                signals.iloc[i] = True
        
        return signals
