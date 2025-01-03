from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
import numpy as np

from .strategy_base import StrategyBase, BacktestConfig

@dataclass
class AMVRConfig:
    """Configuration for AMVR strategy."""
    lookback_period: int = 20
    volatility_window: int = 10
    entry_z_score: float = 1.5
    exit_z_score: float = 0.0
    adaptive_factor: float = 0.1

class AMVRStrategy(StrategyBase):
    """Adaptive Mean-Variance Reversion Strategy.
    
    Implements a mean reversion strategy that adapts to changing market conditions
    by adjusting entry/exit thresholds based on recent volatility.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the strategy with configuration."""
        super().__init__(BacktestConfig(**config['backtest_settings']))
        self.amvr_config = AMVRConfig(**config['strategies']['AMVR'])
    
    def calculate_adaptive_thresholds(self, prices: pd.Series) -> tuple:
        """Calculate adaptive thresholds based on recent volatility."""
        # Calculate returns and volatility
        returns = prices.pct_change()
        volatility = returns.rolling(window=self.amvr_config.volatility_window).std()
        
        # Calculate volatility ratio (current vs historical)
        vol_ratio = (
            volatility / 
            volatility.rolling(window=self.amvr_config.lookback_period).mean()
        )
        
        # Adjust thresholds based on volatility ratio
        adjustment = np.exp(-self.amvr_config.adaptive_factor * (vol_ratio - 1))
        entry_threshold = self.amvr_config.entry_z_score * adjustment
        exit_threshold = self.amvr_config.exit_z_score * adjustment
        
        return entry_threshold, exit_threshold
    
    def calculate_z_score(self, prices: pd.Series) -> pd.Series:
        """Calculate z-score of price relative to its moving average."""
        # Calculate moving average and standard deviation
        ma = prices.rolling(window=self.amvr_config.lookback_period).mean()
        std = prices.rolling(window=self.amvr_config.lookback_period).std()
        
        # Calculate z-score
        z_score = (prices - ma) / std
        return z_score
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on adaptive mean-variance reversion."""
        # Get price data
        prices = data.iloc[:, 0].astype(float)
        
        # Calculate z-score and adaptive thresholds
        z_score = self.calculate_z_score(prices)
        entry_threshold, exit_threshold = self.calculate_adaptive_thresholds(prices)
        
        # Generate signals
        signals = pd.Series(False, index=prices.index)
        
        # Long when z-score is below negative entry threshold (oversold)
        signals[z_score < -entry_threshold] = True
        
        # Exit when z-score is between exit thresholds
        signals[(z_score >= -exit_threshold) & (z_score <= exit_threshold)] = False
        
        # Short when z-score is above positive entry threshold (overbought)
        signals[z_score > entry_threshold] = False
        
        return signals.astype(bool)
