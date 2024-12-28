from strategy_framework import Strategy
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler

class VolatilityRegimeStrategy(Strategy):
    """
    Strategy 5: Volatility Regime & Cross-Asset Strategy
    
    Logic:
    1. Identifies volatility regimes using multiple methods:
       - Realized volatility
       - VIX regime
       - Cross-asset volatility (rates, credit, equity)
    2. Uses volatility surface (term structure) information
    3. Incorporates cross-asset correlations
    4. Adapts position sizing based on risk environment
    """
    
    def __init__(self, df: pd.DataFrame,
                 vol_window: int = 30,
                 correlation_window: int = 90,
                 regime_window: int = 252,
                 vol_threshold: float = 1.2):
        super().__init__(df)
        self.vol_window = vol_window
        self.correlation_window = correlation_window
        self.regime_window = regime_window
        self.vol_threshold = vol_threshold
        
    def _calculate_vol_surface_score(self) -> pd.Series:
        """Calculate volatility surface score using VIX"""
        # VIX represents implied vol
        implied_vol = self.df['vix']
        # Calculate realized vol using different windows
        realized_vols = pd.DataFrame(index=self.df.index)
        
        realized_vols[f'vol_{self.vol_window}'] = self.df[self.target_col] \
            .pct_change().rolling(self.vol_window).std() * np.sqrt(252)
        
        # Calculate vol risk premium (implied - realized)
        vol_premium = implied_vol - realized_vols.mean(axis=1)
        
        # Standardize the premium
        vol_premium_zscore = (vol_premium - vol_premium.rolling(252).mean()) / \
                            vol_premium.rolling(252).std()
                            
        return -vol_premium_zscore  # Negative because high premium = bad
        
    def _calculate_cross_asset_correlation(self) -> pd.Series:
        """Calculate dynamic correlations with other assets"""
        # Calculate returns for target and other assets
        target_returns = self.df[self.target_col].pct_change()
        assets = ['cad_oas', 'us_hy_oas', 'us_ig_oas']
        asset_returns = self.df[assets].pct_change()
        
        # Calculate rolling correlations
        correlations = pd.DataFrame(index=self.df.index)
        for asset in assets:
            correlations[asset] = target_returns.rolling(self.correlation_window) \
                .corr(asset_returns[asset])
        
        # Average correlation
        avg_correlation = correlations.mean(axis=1)
        
        # Standardize
        correlation_zscore = (avg_correlation - avg_correlation.rolling(252).mean()) / \
                           avg_correlation.rolling(252).std()
                           
        return -correlation_zscore  # Negative because high correlation = bad
        
    def _calculate_vol_regime(self) -> pd.Series:
        """Identify volatility regime using multiple indicators"""
        # Calculate volatility for different assets
        assets = ['cad_oas', 'us_hy_oas', 'us_ig_oas']
        vol_indicators = pd.DataFrame(index=self.df.index)
        
        for asset in assets:
            vol = self.df[asset].pct_change().rolling(20).std() * np.sqrt(252)
            vol_indicators[f'{asset}_vol'] = (vol < vol.rolling(252).mean())
            
        # Add VIX regime
        vol_indicators['vix_regime'] = self.df['vix'] < self.df['vix'].rolling(252).mean()
        
        # Combine regimes
        low_vol_regime = vol_indicators.mean(axis=1) > 0.5
        return low_vol_regime
        
    def generate_signals(self) -> pd.Series:
        """Generate trading signals based on volatility regime"""
        # Calculate component scores
        vol_surface_score = self._calculate_vol_surface_score()
        correlation_score = self._calculate_cross_asset_correlation()
        vol_regime = self._calculate_vol_regime()
        
        # Calculate trend strength
        returns = self.df[self.target_col].pct_change()
        trend = returns.rolling(60).mean() / returns.rolling(60).std()
        trend_strength = trend.abs()
        
        # Print regime analysis
        print("\nVolatility Regime Strategy Analysis:")
        print("===================================")
        low_vol_days = vol_regime.sum()
        print(f"Low Volatility Regime: {low_vol_days} days ({low_vol_days/len(vol_regime)*100:.1f}% of time)")
        print(f"Average Trend Strength: {trend_strength.mean():.2f}")
        print(f"Average Correlation Score: {correlation_score.mean():.2f}")
        
        # Generate signals with multiple conditions
        signals = (
            vol_regime &  # Low volatility regime
            (vol_surface_score > 0) &  # Positive vol surface score
            (correlation_score > -0.3) &  # Not too high correlations
            (trend_strength > 0.1)  # Some trend presence
        )
        
        # Apply smoothing to reduce whipsaws
        signals = signals.rolling(5).mean() > 0.6  # Require 3 out of 5 days to be true
        
        return signals
