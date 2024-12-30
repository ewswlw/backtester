from .strategy_framework import Strategy
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class MLEnsembleStrategy(Strategy):
    """
    Strategy 7: Machine Learning Ensemble Strategy
    
    Logic:
    1. Uses an ensemble of machine learning models:
       - Random Forest
       - Logistic Regression
       - Support Vector Machine
    2. Features include:
       - Technical indicators
       - Macro factors
       - Cross-asset signals
    3. Adaptive training windows based on market conditions
    4. Probability-based position sizing
    """
    
    def __init__(self, df: pd.DataFrame,
                 base_window: int = 126,  # Reduced from 252 for more frequent updates
                 prediction_threshold: float = 0.65,  # Increased from 0.6
                 min_train_size: int = 756):  # Increased from 504 for more data
        super().__init__(df)
        self.base_window = base_window
        self.prediction_threshold = prediction_threshold
        self.min_train_size = min_train_size
        
    def _calculate_features(self) -> pd.DataFrame:
        """Calculate technical and macro features"""
        price = self.df[self.target_col]
        features = pd.DataFrame(index=self.df.index)
        
        # Technical indicators
        features['ma_20'] = price.rolling(20).mean() / price - 1
        features['ma_60'] = price.rolling(60).mean() / price - 1
        features['rsi'] = self._calculate_rsi(price, 14)
        features['vol_20'] = price.pct_change().rolling(20).std()
        features['vol_60'] = price.pct_change().rolling(60).std()
        
        # Macro factors
        features['us_economic_regime'] = self.df['us_economic_regime']
        
        # Cross-asset features
        for col in ['cad_oas', 'us_hy_oas', 'us_ig_oas']:
            features[f'{col}_zscore'] = (
                self.df[col] - self.df[col].rolling(60).mean()
            ) / self.df[col].rolling(60).std()
            
        return features.fillna(0)
        
    def _calculate_rsi(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Neutral RSI for NaN values
        
    def _prepare_training_data(self, features: pd.DataFrame, 
                             returns: pd.Series, 
                             start_idx: int, 
                             end_idx: int) -> tuple:
        """Prepare training data for ML models"""
        # Get training data
        X_train = features.iloc[start_idx:end_idx]
        
        # Calculate forward returns and create labels
        forward_returns = returns.shift(-1).iloc[start_idx:end_idx]
        y_train = (forward_returns > 0).astype(int)
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        )
        
        return X_train_scaled, y_train, scaler
        
    def _train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> list:
        """Train ensemble of models"""
        # Initialize models with optimized parameters
        rf = RandomForestClassifier(
            n_estimators=200,  # Increased from 100
            max_depth=4,  # Increased from 3
            min_samples_leaf=10,  # Added min samples leaf
            random_state=42
        )
        lr = LogisticRegression(
            C=0.1,  # Added regularization
            class_weight='balanced',  # Added class weights
            random_state=42
        )
        svm = SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            class_weight='balanced',  # Added class weights
            random_state=42
        )
        
        # Train models
        models = []
        for model in [rf, lr, svm]:
            model.fit(X_train, y_train)
            models.append(model)
            
        return models
        
    def _get_ensemble_probabilities(self, models: list, 
                                  X: pd.DataFrame, 
                                  scaler: StandardScaler) -> pd.Series:
        """Get averaged probability predictions from ensemble"""
        # Scale features
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            index=X.index,
            columns=X.columns
        )
        
        # Get probabilities from each model
        probabilities = []
        for model in models:
            prob = model.predict_proba(X_scaled)[:, 1]  # Probability of class 1
            probabilities.append(prob)
            
        # Average probabilities
        avg_prob = np.mean(probabilities, axis=0)
        
        return pd.Series(avg_prob, index=X.index)
        
    def generate_signals(self) -> pd.Series:
        """Generate trading signals using ML ensemble"""
        signals = pd.Series(False, index=self.df.index)
        
        # Calculate features
        features = self._calculate_features()
        returns = self.df[self.target_col].pct_change()
        
        # Initialize tracking variables
        n_trades = 0
        avg_confidence = 0
        
        # Walk-forward training and prediction
        for i in range(self.min_train_size, len(self.df), self.base_window):
            # Define training period
            train_start = max(0, i - 2 * self.base_window)
            train_end = i
            
            # Prepare training data
            X_train, y_train, scaler = self._prepare_training_data(
                features, returns, train_start, train_end
            )
            
            # Train models
            models = self._train_models(X_train, y_train)
            
            # Define prediction period
            pred_start = i
            pred_end = min(i + self.base_window, len(self.df))
            
            # Get predictions for the period
            X_pred = features.iloc[pred_start:pred_end]
            probabilities = self._get_ensemble_probabilities(models, X_pred, scaler)
            
            # Generate binary signals based on probability threshold
            period_signals = probabilities > self.prediction_threshold
            
            # Update signals
            signals.iloc[pred_start:pred_end] = period_signals
            
            # Update statistics
            n_trades += period_signals.sum()
            avg_confidence += probabilities[period_signals].mean() if period_signals.any() else 0
        
        # Print analysis
        print("\nML Ensemble Strategy Analysis:")
        print("=============================")
        print(f"Total Trading Days: {signals.sum()}")
        print(f"Average Prediction Confidence: {avg_confidence / max(1, n_trades):.2%}")
        
        return signals
