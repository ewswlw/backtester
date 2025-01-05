from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Dict, Any

from .strategy_base import StrategyBase, BacktestConfig

@dataclass
class TrendRiskConfig:
    """Configuration for Trend Risk strategy"""
    trend_window: int = 3  # 3-month base window
    vol_window: int = 2  # 2-month volatility window
    vol_target: float = 0.10  # Increased from 0.08 to 0.10
    max_leverage: float = 1.0
    min_holding_period: int = 1  # 1-month minimum holding
    risk_threshold: float = 0.85  # Increased from 0.7 to 0.85
    trend_threshold: float = 0.2  # Reduced from 0.5 to 0.2

class TrendRiskStrategy(StrategyBase):
    """
    A trend-following strategy with dynamic risk management.
    
    Key features:
    1. Trend following on excess returns and credit spreads
    2. Dynamic volatility targeting
    3. Risk-off triggers based on market stress
    4. Momentum confirmation
    """
    
    def __init__(self, config: BacktestConfig):
        """Initialize strategy with configuration."""
        super().__init__(config)
        self.trend_config = TrendRiskConfig()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on trend and risk."""
        if isinstance(data, pd.Series):
            price_series = data
        else:
            price_series = data.iloc[:, 0]  # Use first column as price series
            
        # Calculate simple moving averages
        ma_fast = price_series.rolling(window=42).mean()  # 2-month MA
        ma_slow = price_series.rolling(window=84).mean()  # 4-month MA
        
        # Calculate momentum
        returns = price_series.pct_change()
        mom_1m = returns.rolling(window=21).mean() * 252  # 1-month momentum
        
        # Calculate volatility
        vol = returns.rolling(window=21).std() * np.sqrt(252)  # Annualized volatility
        
        # Generate signals
        signals = pd.Series(False, index=price_series.index)
        
        # Long signal when fast MA crosses above slow MA and momentum is positive
        signals[(ma_fast > ma_slow) & (mom_1m > 0)] = True
        
        # Risk filter: no positions when volatility is too high
        high_vol = vol > 0.20  # More permissive volatility threshold
        signals[high_vol] = False
        
        # Apply minimum holding period (10 days)
        for i in range(10, len(signals)):  # Shorter holding period
            if signals.iloc[i-10:i].any() and not high_vol.iloc[i]:
                signals.iloc[i] = True
        
        return signals
