from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
import numpy as np

from .strategy_base import StrategyBase, BacktestConfig

@dataclass
class RSMRConfig:
    """Configuration for RSMR strategy."""
    lookback_period: int = 20
    entry_z_score: float = 1.5
    exit_z_score: float = 0.0

class RSMRStrategy(StrategyBase):
    """Relative Strength Mean Reversion Strategy.
    
    Goes long when price is significantly below its moving average (oversold)
    and short when significantly above (overbought), based on z-score.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the strategy with configuration."""
        super().__init__(BacktestConfig(**config['backtest_settings']))
        self.rsmr_config = RSMRConfig(**config['strategies']['RSMR'])
    
    def calculate_z_score(self, prices: pd.Series) -> pd.Series:
        """Calculate z-score of price relative to its moving average."""
        # Calculate moving average and standard deviation
        ma = prices.rolling(window=self.rsmr_config.lookback_period, min_periods=1).mean()
        std = prices.rolling(window=self.rsmr_config.lookback_period, min_periods=1).std()
        
        # Calculate z-score
        z_score = (prices - ma) / std
        return z_score
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on relative strength mean reversion."""
        # Get price data
        prices = data.iloc[:, 0].astype(float)
        
        # Calculate z-score
        z_score = self.calculate_z_score(prices)
        
        # Generate signals
        signals = pd.Series(False, index=prices.index)
        
        # Long when oversold (z-score < -entry_threshold)
        signals[z_score < -self.rsmr_config.entry_z_score] = True
        
        # Exit when z-score crosses exit threshold
        signals[(z_score >= -self.rsmr_config.exit_z_score) & 
               (z_score <= self.rsmr_config.exit_z_score)] = False
        
        # Short when overbought (z-score > entry_threshold)
        signals[z_score > self.rsmr_config.entry_z_score] = False
        
        return signals.astype(bool)
