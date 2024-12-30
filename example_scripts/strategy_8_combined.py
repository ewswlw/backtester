from .strategy_framework import Strategy
import pandas as pd
import numpy as np
from .strategy_4_multi_factor import MultiFactorStrategy
from .strategy_5_volatility_regime import VolatilityRegimeStrategy
from .strategy_6_adaptive_trend import AdaptiveTrendStrategy
from .strategy_7_ml_ensemble import MLEnsembleStrategy

class CombinedStrategy(Strategy):
    """
    Strategy 8: Combined Strategy
    
    Logic:
    1. Uses signals from multiple strategies:
       - AdaptiveTrend: For strong trend detection
       - MLEnsemble: For regime prediction
       - VolatilityRegime: For risk management
       - MultiFactor: For fundamental backdrop
    2. Combines signals based on strategy strengths
    3. Uses volatility and correlation for risk management
    """
    
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        # Initialize component strategies
        self.trend_strategy = AdaptiveTrendStrategy(df)
        self.ml_strategy = MLEnsembleStrategy(df)
        self.vol_strategy = VolatilityRegimeStrategy(df)
        self.factor_strategy = MultiFactorStrategy(df)
        
    def _calculate_strategy_weights(self) -> pd.DataFrame:
        """Calculate dynamic strategy weights based on market conditions"""
        weights = pd.DataFrame(index=self.df.index)
        
        # Base weights
        weights['trend'] = 0.3
        weights['ml'] = 0.3
        weights['vol'] = 0.2
        weights['factor'] = 0.2
        
        # Adjust weights based on VIX
        high_vol = self.df['vix'] > self.df['vix'].rolling(252).mean()
        weights.loc[high_vol, ['trend', 'ml']] *= 0.8
        weights.loc[high_vol, ['vol', 'factor']] *= 1.2
        
        # Adjust weights based on economic regime
        strong_regime = self.df['us_economic_regime'] > 0.7
        weights.loc[strong_regime, ['trend', 'factor']] *= 1.2
        weights.loc[strong_regime, ['ml', 'vol']] *= 0.8
        
        # Normalize weights to sum to 1
        row_sums = weights.sum(axis=1)
        weights = weights.div(row_sums, axis=0)
        
        return weights
        
    def generate_signals(self) -> pd.Series:
        """Generate trading signals using combined approach"""
        # Get individual strategy signals
        trend_signals = self.trend_strategy.generate_signals()
        ml_signals = self.ml_strategy.generate_signals()
        vol_signals = self.vol_strategy.generate_signals()
        factor_signals = self.factor_strategy.generate_signals()
        
        # Calculate dynamic weights
        weights = self._calculate_strategy_weights()
        
        # Combine signals with weights
        combined_score = (
            weights['trend'] * trend_signals.astype(float) +
            weights['ml'] * ml_signals.astype(float) +
            weights['vol'] * vol_signals.astype(float) +
            weights['factor'] * factor_signals.astype(float)
        )
        
        # Generate final signals
        signals = combined_score > 0.5  # Require majority agreement
        
        # Print analysis
        print("\nCombined Strategy Analysis:")
        print("==========================")
        print("Strategy Agreement Analysis:")
        agreement = pd.DataFrame({
            'trend': trend_signals,
            'ml': ml_signals,
            'vol': vol_signals,
            'factor': factor_signals
        })
        agreement_count = agreement.sum(axis=1)
        for i in range(5):
            count = (agreement_count == i).sum()
            print(f"{i} strategies agree: {count} days ({count/len(signals)*100:.1f}% of time)")
        
        signal_days = signals.sum()
        print(f"\nDays in Market: {signal_days} ({signal_days/len(signals)*100:.1f}% of time)")
        print(f"Number of Trades: {(signals != signals.shift(1)).sum()}")
        
        # Print average weights
        print("\nAverage Strategy Weights:")
        for col in weights.columns:
            print(f"{col}: {weights[col].mean():.2f}")
        
        return signals
