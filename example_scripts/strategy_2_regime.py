from .strategy_framework import Strategy
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class MacroRegimeStrategy(Strategy):
    """
    Strategy 2: Macro Regime-Based Strategy
    
    Logic:
    - Uses multiple macro indicators to identify favorable regimes
    - Combines VIX, yield curve, and economic surprise indicators
    - Employs z-score standardization for regime identification
    """
    
    def __init__(self, df: pd.DataFrame, 
                 lookback: int = 252,
                 z_threshold: float = 1.0):
        super().__init__(df)
        self.lookback = lookback
        self.z_threshold = z_threshold
        
    def generate_signals(self) -> pd.Series:
        signals = pd.Series(False, index=self.df.index)
        
        # 1. VIX regime
        vix_zscore = self._calculate_zscore('vix')
        vix_regime = vix_zscore < self.z_threshold
        
        # 2. Yield curve regime
        curve_zscore = self._calculate_zscore('us_3m_10y')
        curve_regime = curve_zscore > -self.z_threshold
        
        # 3. Economic surprises regime
        surprise_indicators = ['us_growth_surprises', 'us_inflation_surprises', 
                             'us_hard_data_surprises']
        surprise_regimes = []
        
        for indicator in surprise_indicators:
            if indicator in self.df.columns:
                zscore = self._calculate_zscore(indicator)
                surprise_regimes.append(zscore > 0)
        
        combined_surprise_regime = pd.concat(surprise_regimes, axis=1).mean(axis=1) > 0.5
        
        # 4. Economic regime check
        if 'us_economic_regime' in self.df.columns:
            econ_regime = self.df['us_economic_regime'] > 0
        else:
            econ_regime = pd.Series(True, index=self.df.index)
        
        # Combine all regime signals
        signals = (vix_regime & 
                  curve_regime & 
                  combined_surprise_regime &
                  econ_regime)
        
        # Add momentum filter
        momentum = self.df[self.target_col].pct_change(60) > 0
        
        return (signals & momentum).fillna(False)
    
    def _calculate_zscore(self, column: str) -> pd.Series:
        """Calculate rolling z-score for a given column"""
        if column not in self.df.columns:
            return pd.Series(0, index=self.df.index)
            
        rolling_mean = self.df[column].rolling(self.lookback).mean()
        rolling_std = self.df[column].rolling(self.lookback).std()
        
        zscore = (self.df[column] - rolling_mean) / rolling_std
        return zscore.fillna(0)
