from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
import numpy as np

from .strategy_base import StrategyBase, BacktestConfig

@dataclass
class MTTFConfig:
    """Configuration for MTTF strategy."""
    lookback_period: int = 20
    volatility_window: int = 10
    entry_threshold: float = 1.0
    exit_threshold: float = 0.0

class MTTFStrategy(StrategyBase):
    """Mean Time To Failure Strategy.
    
    Uses volatility-adjusted time series to identify potential market turning points
    based on the concept of mean time between failures in reliability engineering.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the strategy with configuration."""
        super().__init__(BacktestConfig(**config['backtest_settings']))
        self.mttf_config = MTTFConfig(**config['strategies']['MTTF'])
    
    def calculate_mttf_score(self, prices: pd.Series) -> pd.Series:
        """Calculate MTTF score based on volatility-adjusted returns."""
        # Calculate returns and volatility
        returns = prices.pct_change()
        volatility = returns.rolling(window=self.mttf_config.volatility_window).std()
        
        # Calculate volatility-adjusted returns
        vol_adj_returns = returns / volatility
        
        # Calculate MTTF score (cumulative sum of vol-adjusted returns)
        mttf_score = vol_adj_returns.rolling(window=self.mttf_config.lookback_period).sum()
        
        return mttf_score
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on MTTF score."""
        # Get price data
        prices = data.iloc[:, 0].astype(float)
        
        # Calculate MTTF score
        mttf_score = self.calculate_mttf_score(prices)
        
        # Generate signals
        signals = pd.Series(False, index=prices.index)
        
        # Long when MTTF score is below negative threshold (oversold)
        signals[mttf_score < -self.mttf_config.entry_threshold] = True
        
        # Exit positions when MTTF score is between thresholds
        signals[(mttf_score >= -self.mttf_config.exit_threshold) & 
               (mttf_score <= self.mttf_config.exit_threshold)] = False
        
        # Short when MTTF score is above positive threshold (overbought)
        signals[mttf_score > self.mttf_config.entry_threshold] = False
        
        return signals.astype(bool)
