from abc import ABC, abstractmethod
import pandas as pd
import vectorbt as vbt
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class StrategyConfig:
    """Base configuration for all strategies"""
    rebalance_freq: str = 'M'  # Default monthly rebalancing
    initial_capital: float = 100.0
    size: float = 1.0
    size_type: str = 'percent'

@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    start_date: str
    end_date: str
    rebalance_freq: str
    initial_capital: float
    size: float
    size_type: str

class StrategyBase(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, config: BacktestConfig):
        """Initialize strategy with configuration."""
        self.config = config
        self._portfolio = None
        self._signals = None
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals for the strategy.
        
        Args:
            data (pd.DataFrame): Price and other market data
            
        Returns:
            pd.Series: Boolean series with True for long positions
        """
        pass
    
    def backtest(self, price_series: pd.Series) -> vbt.Portfolio:
        """Run backtest for the strategy"""
        # Generate signals
        self._signals = self.generate_signals(price_series.to_frame())
        
        # Convert to pandas Series if DataFrame is returned
        if isinstance(self._signals, pd.DataFrame):
            self._signals = self._signals.iloc[:, 0]
        
        # Ensure boolean type
        self._signals = self._signals.astype(bool)
        
        # Generate entry/exit signals
        shifted_signals = pd.Series(False, index=self._signals.index)  # Initialize with False
        shifted_signals[1:] = self._signals[:-1]  # Copy values without using fillna
        
        entries = self._signals & ~shifted_signals
        exits = ~self._signals & shifted_signals
        
        # Create portfolio
        portfolio = vbt.Portfolio.from_signals(
            price_series,
            entries=entries,
            exits=exits,
            init_cash=self.config.initial_capital,
            size=self.config.size,
            size_type=self.config.size_type,
            freq='1D'  # Always use daily frequency
        )
        
        return portfolio
    
    def get_stats(self) -> Dict[str, Any]:
        """Get strategy performance statistics.
        
        Returns:
            Dict[str, Any]: Dictionary of performance metrics
        """
        if self._portfolio is None:
            raise ValueError("Must run backtest before getting stats")
        return self._portfolio.stats()
