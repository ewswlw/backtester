import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

from .strategy_base import StrategyBase, BacktestConfig

@dataclass
class MAConfig:
    """Configuration for MA strategy."""
    ma_window: int = 20
    entry_threshold: float = 0.0

class MovingAverageStrategy(StrategyBase):
    """Moving average crossover strategy."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MA strategy."""
        super().__init__(BacktestConfig(**config['backtest_settings']))
        self.ma_config = MAConfig(**config['strategies']['MA'])
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on moving average crossover."""
        # Get price data
        prices = data.iloc[:, 0].astype(float)
        
        # Calculate moving average
        ma = prices.rolling(window=self.ma_config.ma_window, min_periods=1).mean()
        
        # Generate signals
        signals = pd.Series(False, index=prices.index)
        
        # Long when price above MA, short when below
        signals[prices > (ma + self.ma_config.entry_threshold)] = True
        signals[prices < (ma - self.ma_config.entry_threshold)] = False
        
        # Fill any missing values with previous signal
        signals = signals.astype(bool)
        return signals
