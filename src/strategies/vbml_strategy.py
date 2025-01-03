import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .strategy_base import StrategyBase, StrategyConfig

@dataclass
class VBMLConfig(StrategyConfig):
    """Configuration for Volatility Breakout with Machine Learning strategy"""
    # Volatility parameters
    bb_window: int = 20
    bb_std: float = 2.0
    atr_window: int = 14
    
    # Breakout parameters
    breakout_threshold: float = 1.0
    
    # Machine Learning parameters
    feature_window: int = 20
    training_window: int = 252  # About 1 year
    prediction_threshold: float = 0.6
    model_params: dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'random_state': 42,
        'max_depth': 5,
        'min_samples_split': 5
    })

class VolatilityBreakoutMLStrategy(StrategyBase):
    """
    Volatility Breakout with Machine Learning (VBML) Strategy
    
    This strategy combines traditional volatility breakout signals with
    machine learning predictions for high-probability breakout patterns.
    """
    
    def __init__(self, config: Optional[VBMLConfig] = None):
        super().__init__(config or VBMLConfig())
        self.config: VBMLConfig = self.config
        self._model = RandomForestClassifier(**self.config.model_params)
        self._scaler = StandardScaler()
        
    def _calculate_volatility_indicators(self, prices: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Calculate Bollinger Bands and ATR"""
        # Calculate Bollinger Bands
        rolling_mean = prices.rolling(window=self.config.bb_window).mean()
        rolling_std = prices.rolling(window=self.config.bb_window).std()
        
        bb = pd.DataFrame({
            'middle': rolling_mean,
            'upper': rolling_mean + (rolling_std * self.config.bb_std),
            'lower': rolling_mean - (rolling_std * self.config.bb_std)
        })
        
        # Calculate ATR
        high = prices
        low = prices
        close = prices
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.config.atr_window).mean()
        
        return bb, atr
    
    def _create_features(self, prices: pd.Series, bb: pd.DataFrame, atr: pd.Series) -> pd.DataFrame:
        """Create features for machine learning model"""
        features = pd.DataFrame(index=prices.index)
        
        # Price relative to Bollinger Bands
        features['bb_position'] = (prices - bb['middle']) / (bb['upper'] - bb['middle'])
        
        # Volatility features
        features['atr_ratio'] = atr / prices
        features['bb_width'] = (bb['upper'] - bb['lower']) / bb['middle']
        
        # Momentum features
        for window in [5, 10, 20]:
            # Returns
            features[f'return_{window}d'] = prices.pct_change(window)
            
            # Volatility
            features[f'vol_{window}d'] = prices.pct_change().rolling(window).std()
            
            # Rate of change
            features[f'roc_{window}d'] = (prices - prices.shift(window)) / prices.shift(window)
        
        return features
    
    def _create_labels(self, prices: pd.Series, forward_window: int = 20) -> pd.Series:
        """Generate labels for training.
        
        Args:
            prices (pd.Series): Price series
            forward_window (int, optional): Forward window for returns. Defaults to 20.
            
        Returns:
            pd.Series: Binary labels indicating breakout opportunities
        """
        # Calculate forward returns
        forward_returns = prices.shift(-forward_window) / prices - 1
        
        # Calculate volatility threshold
        vol_threshold = self.config.breakout_threshold * prices.pct_change().std()
        
        # Generate binary labels
        labels = (forward_returns > vol_threshold).shift(forward_window)
        
        # Fill NaN values with False before converting to int
        labels = labels.fillna(False)
        
        return labels.astype(int)  # Convert to integers (0 or 1)
    
    def _train_model(self, X: pd.DataFrame, y: pd.Series, start_idx: int, end_idx: int) -> None:
        """Train the model on the data from start_idx to end_idx.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            start_idx (int): Start index for training
            end_idx (int): End index for training
        """
        # Get training data
        X_train = X.iloc[start_idx:end_idx]
        y_train = y.iloc[start_idx:end_idx]
        
        # Drop NaN values
        mask = ~(X_train.isna().any(axis=1) | y_train.isna())
        X_train = X_train[mask]
        y_train = y_train[mask]
        
        if len(X_train) == 0 or len(y_train) == 0:
            return
        
        # Scale features
        X_scaled = self._scaler.fit_transform(X_train)
        
        # Train model
        self._model.fit(X_scaled, y_train)
    
    def _predict_breakout(self, features: pd.DataFrame, current_idx: int) -> float:
        """Predict breakout probability"""
        if current_idx < self.config.training_window:
            return 0.0
        
        # Scale current features
        current_features = features.iloc[current_idx].values.reshape(1, -1)
        X_scaled = self._scaler.transform(current_features)
        
        # Get prediction probability
        return self._model.predict_proba(X_scaled)[0][1]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals using machine learning predictions.
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            pd.Series: Trading signals
        """
        prices = data.iloc[:, 0]
        signals = pd.Series(index=data.index, dtype=float)
        
        # Generate features and labels
        bb, atr = self._calculate_volatility_indicators(prices)
        features = self._create_features(prices, bb, atr)
        labels = self._create_labels(prices)
        
        # Use expanding window with monthly retraining
        lookback = 252  # 1 year of data
        step_size = 20  # Retrain monthly
        
        for i in range(lookback, len(data), step_size):
            start_idx = max(0, i - lookback)
            end_idx = i
            
            # Train model on historical data
            self._train_model(features, labels, start_idx, end_idx)
            
            # Make predictions for next period
            if hasattr(self, '_model'):
                X_pred = features.iloc[end_idx:min(end_idx + step_size, len(data))]
                if len(X_pred) > 0:
                    X_pred_scaled = self._scaler.transform(X_pred)
                    preds = self._model.predict(X_pred_scaled)
                    signals.iloc[end_idx:min(end_idx + step_size, len(data))] = preds
        
        # Forward fill any remaining NaN values
        return signals.ffill().fillna(0)
