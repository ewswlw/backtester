from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
import numpy as np

from .strategy_base import StrategyBase, BacktestConfig

@dataclass
class MRSAConfig:
    """Configuration for MRSA strategy."""
    lookback_period: int = 20
    sentiment_window: int = 10
    entry_threshold: float = 1.5
    exit_threshold: float = 0.0

class MRSAStrategy(StrategyBase):
    """Mean Reversion Sentiment Analysis Strategy.
    
    Combines mean reversion with a sentiment indicator based on
    price momentum and volatility patterns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the strategy with configuration."""
        super().__init__(BacktestConfig(**config['backtest_settings']))
        self.mrsa_config = MRSAConfig(**config['strategies']['MRSA'])
    
    def calculate_sentiment_score(self, prices: pd.Series) -> pd.Series:
        """Calculate sentiment score based on price action patterns."""
        # Calculate returns and volatility
        returns = prices.pct_change()
        volatility = returns.rolling(window=self.mrsa_config.sentiment_window).std()
        
        # Calculate momentum
        momentum = returns.rolling(window=self.mrsa_config.sentiment_window).mean()
        
        # Calculate sentiment score combining momentum and volatility
        sentiment_score = (momentum / volatility).rolling(
            window=self.mrsa_config.lookback_period
        ).mean()
        
        return sentiment_score
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on sentiment score."""
        # Get price data
        prices = data.iloc[:, 0].astype(float)
        
        # Calculate sentiment score
        sentiment_score = self.calculate_sentiment_score(prices)
        
        # Generate signals
        signals = pd.Series(False, index=prices.index)
        
        # Long when sentiment is strongly negative (oversold)
        signals[sentiment_score < -self.mrsa_config.entry_threshold] = True
        
        # Exit positions when sentiment is neutral
        signals[(sentiment_score >= -self.mrsa_config.exit_threshold) & 
               (sentiment_score <= self.mrsa_config.exit_threshold)] = False
        
        # Short when sentiment is strongly positive (overbought)
        signals[sentiment_score > self.mrsa_config.entry_threshold] = False
        
        return signals.astype(bool)
