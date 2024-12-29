from .strategy_framework import Strategy
import pandas as pd

class BuyAndHoldStrategy(Strategy):
    """
    Strategy 0: Buy and Hold Strategy
    
    This is the baseline strategy that simply buys and holds the target asset.
    Used as a benchmark for comparing other strategies.
    
    Logic:
    - Always generates a True signal (always invested)
    - No entry/exit rules
    - No position sizing
    - No risk management
    """
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'cad_ig_er_ytd_index'):
        super().__init__(df, target_col)
        
    def generate_signals(self) -> pd.Series:
        """Generate constant True signals for buy and hold"""
        return pd.Series(True, index=self.df.index)
    
    def backtest(self) -> dict:
        """
        Run backtest for buy and hold strategy.
        Overrides parent method to ensure consistent benchmark calculation.
        """
        signals = self.generate_signals()
        returns = self._calculate_returns(signals)
        metrics = self._calculate_metrics(returns, signals)
        
        # Add strategy-specific metrics
        metrics.update({
            'strategy_type': 'Benchmark',
            'avg_position_size': 1.0,  # Always fully invested
            'trading_frequency': 'Never',  # Only trades once at start
            'risk_management': 'None',  # No active risk management
            'signal_type': 'Static'  # Signals don't change
        })
        
        return metrics
