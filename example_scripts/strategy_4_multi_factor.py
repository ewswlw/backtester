from strategy_framework import Strategy
import pandas as pd
import numpy as np

class MultiFactorStrategy(Strategy):
    """
    Strategy 4: Multi-Factor Strategy
    
    Logic:
    1. Combines multiple factors:
       - Value (relative spreads)
       - Momentum (price trends)
       - Volatility (risk regime)
       - Macro (economic conditions)
    2. Dynamic factor weights based on market regime
    3. Minimum holding period to reduce turnover
    """
    
    def __init__(self, df: pd.DataFrame,
                 momentum_window: int = 60,
                 vol_window: int = 30,
                 regime_window: int = 252,
                 zscore_threshold: float = 0.3,  # Lowered threshold
                 min_holding_period: int = 5):   # Shorter holding period
        super().__init__(df)
        self.momentum_window = momentum_window
        self.vol_window = vol_window
        self.regime_window = regime_window
        self.zscore_threshold = zscore_threshold
        self.min_holding_period = min_holding_period
        
    def _calculate_value_score(self) -> pd.Series:
        """Calculate value score based on relative spreads"""
        # Calculate z-scores for each spread
        spread_cols = ['cad_oas', 'us_hy_oas', 'us_ig_oas']
        zscores = pd.DataFrame(index=self.df.index)
        
        for col in spread_cols:
            rolling_mean = self.df[col].rolling(self.regime_window).mean()
            rolling_std = self.df[col].rolling(self.regime_window).std()
            zscores[col] = -(self.df[col] - rolling_mean) / rolling_std
            
        # Average z-scores (negative because higher spreads = worse value)
        return zscores.mean(axis=1)
        
    def _calculate_momentum_score(self) -> pd.Series:
        """Calculate momentum score using multiple timeframes"""
        price = self.df[self.target_col]
        
        # Calculate returns over different windows
        returns = pd.DataFrame(index=self.df.index)
        for window in [20, 60, 120]:
            returns[f'ret_{window}d'] = price.pct_change(window)
            
        # Z-score the returns
        zscores = returns.apply(lambda x: (x - x.rolling(self.regime_window).mean()) / 
                                         x.rolling(self.regime_window).std())
        
        return zscores.mean(axis=1)
        
    def _calculate_volatility_score(self) -> pd.Series:
        """Calculate volatility score"""
        # Calculate rolling volatility
        returns = self.df[self.target_col].pct_change()
        vol = returns.rolling(self.vol_window).std() * np.sqrt(252)
        
        # Compare to VIX
        vol_ratio = self.df['vix'] / (vol * 100)  # VIX is in percentage points
        
        # Convert to z-score
        vol_zscore = (vol_ratio - vol_ratio.rolling(self.regime_window).mean()) / \
                     vol_ratio.rolling(self.regime_window).std()
                     
        return -vol_zscore  # Negative because higher vol is bad
        
    def _calculate_macro_score(self) -> pd.Series:
        """Calculate macro score using multiple indicators"""
        macro_cols = ['us_growth_surprises', 'us_hard_data_surprises', 
                     'us_lei_yoy', 'us_3m_10y']
        
        # Calculate z-scores for each indicator
        scores = []
        for col in macro_cols:
            mean = self.df[col].rolling(self.regime_window).mean()
            std = self.df[col].rolling(self.regime_window).std()
            score = (self.df[col] - mean) / std
            scores.append(score)
            
        return pd.concat(scores, axis=1).mean(axis=1)
        
    def generate_signals(self) -> pd.Series:
        """Generate trading signals based on multi-factor model"""
        # Calculate factor scores
        value_score = self._calculate_value_score()
        momentum_score = self._calculate_momentum_score()
        vol_score = self._calculate_volatility_score()
        macro_score = self._calculate_macro_score()
        
        # Print factor analysis
        print("\nMulti-Factor Strategy Analysis:")
        print("==============================")
        print(f"Average Value Score: {value_score.mean():.2f}")
        print(f"Average Momentum Score: {momentum_score.mean():.2f}")
        print(f"Average Volatility Score: {vol_score.mean():.2f}")
        print(f"Average Macro Score: {macro_score.mean():.2f}")
        
        # Dynamic factor weights based on regime
        is_risk_on = (self.df['us_economic_regime'] > 0.7) & \
                     (self.df['vix'] < self.df['vix'].rolling(252).mean())
        
        # More aggressive weights in risk-on regime
        weights = pd.DataFrame(index=self.df.index)
        weights['value'] = np.where(is_risk_on, 0.35, 0.15)
        weights['momentum'] = np.where(is_risk_on, 0.35, 0.15)
        weights['volatility'] = np.where(is_risk_on, 0.15, 0.35)
        weights['macro'] = np.where(is_risk_on, 0.15, 0.35)
        
        # Calculate combined score with regime-based normalization
        combined_score = pd.Series(0.0, index=self.df.index)
        for regime in [True, False]:
            regime_mask = is_risk_on == regime
            if regime_mask.any():
                regime_score = (weights.loc[regime_mask, 'value'] * value_score[regime_mask] +
                              weights.loc[regime_mask, 'momentum'] * momentum_score[regime_mask] +
                              weights.loc[regime_mask, 'volatility'] * vol_score[regime_mask] +
                              weights.loc[regime_mask, 'macro'] * macro_score[regime_mask])
                # Normalize within regime
                regime_score = (regime_score - regime_score.mean()) / regime_score.std()
                combined_score[regime_mask] = regime_score
        
        # Generate entry signals
        entry_signals = (combined_score > self.zscore_threshold) & \
                       (combined_score > combined_score.shift(1))
        
        # Generate exit signals
        exit_signals = (combined_score < -self.zscore_threshold) | \
                      (combined_score < combined_score.shift(1))
        
        # Initialize position signals
        signals = pd.Series(False, index=self.df.index)
        position = False
        
        # Apply signals with minimum holding period
        for i in range(1, len(signals)):
            if not position and entry_signals.iloc[i]:
                position = True
                signals.iloc[i] = True
            elif position and exit_signals.iloc[i] and i > self.min_holding_period:
                position = False
                signals.iloc[i] = False
            else:
                signals.iloc[i] = position
        
        # Print signal analysis
        signal_days = signals.sum()
        print(f"Days in Market: {signal_days} ({signal_days/len(signals)*100:.1f}% of time)")
        print(f"Risk-On Days: {is_risk_on.sum()} ({is_risk_on.sum()/len(is_risk_on)*100:.1f}% of time)")
        print(f"Number of Trades: {(signals != signals.shift(1)).sum()}")
        
        return signals
