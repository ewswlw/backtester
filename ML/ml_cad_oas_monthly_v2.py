import numpy as np
import pandas as pd
import logging
import optuna
import warnings
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import ElasticNet, QuantileRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from hmmlearn import hmm
from hmmlearn.hmm import GaussianHMM
from typing import Dict, List, Tuple, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from statsmodels.stats.multitest import multipletests
from arch import arch_model
import traceback
from scipy import stats
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KalmanFeatureEstimator:
    """Estimates time-varying relationships between features using Kalman Filter"""
    
    def __init__(self, n_states: int, dt: float = 1.0):
        self.kf = KalmanFilter(dim_x=n_states, dim_z=1)
        self.dt = dt
        self._setup_filter()
        
    def _setup_filter(self):
        """Initialize Kalman Filter parameters"""
        # State transition matrix
        self.kf.F = np.eye(self.kf.dim_x)
        
        # Measurement matrix
        self.kf.H = np.ones((1, self.kf.dim_x))
        
        # Covariance matrices
        self.kf.P *= 1000.  # Initial state uncertainty
        self.kf.R = 5.0     # Measurement uncertainty
        
        # Process noise
        self.kf.Q = Q_discrete_white_noise(dim=self.kf.dim_x, dt=self.dt, var=0.1)
        
    def update(self, measurement: float):
        """Update state estimate with new measurement"""
        self.kf.predict()
        self.kf.update(measurement)
        return self.kf.x
        
    def get_state(self) -> np.ndarray:
        """Get current state estimate"""
        return self.kf.x.copy()

class GARCHFeatureGenerator:
    """Generate GARCH-based volatility features"""
    
    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self.models = {}
        
    def fit_predict(self, series: pd.Series, window: int = 252) -> pd.Series:
        """Fit GARCH model and predict volatility"""
        vol_forecast = pd.Series(index=series.index, dtype=float)
        
        for i in range(window, len(series)):
            train_data = series[i-window:i]
            try:
                model = arch_model(train_data, p=self.p, q=self.q)
                res = model.fit(disp='off')
                vol_forecast.iloc[i] = res.forecast().variance.values[-1]
            except:
                vol_forecast.iloc[i] = np.nan
                
        return vol_forecast.fillna(method='ffill')

class MarketRegimeClassifier:
    """Classify market regimes using HMM and economic indicators"""
    
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.hmm = hmm.GaussianHMM(n_components=n_regimes, random_state=42)
        self.scaler = StandardScaler()
        
    def fit_predict(self, features: pd.DataFrame) -> np.ndarray:
        """Fit HMM and predict regimes"""
        X = self.scaler.fit_transform(features)
        return self.hmm.fit_predict(X)

class CrossAssetFeatureGenerator:
    """Generate cross-asset relationship features"""
    
    def __init__(self, windows=[20, 60, 120]):
        self.windows = windows
        
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate correlation and lead-lag features"""
        features = pd.DataFrame(index=df.index)
        
        # Rolling correlations
        for w in self.windows:
            for col1 in df.columns:
                for col2 in df.columns:
                    if col1 < col2:
                        corr = df[col1].rolling(w).corr(df[col2])
                        features[f'corr_{col1}_{col2}_{w}d'] = corr
                        
        # Lead-lag relationships
        for w in self.windows:
            for col in df.columns:
                features[f'lead_{col}_{w}d'] = df[col].shift(-w)
                features[f'lag_{col}_{w}d'] = df[col].shift(w)
                
        return features

class TwoStageModel:
    """Two-stage model with regime prediction and conditional spread prediction"""
    
    def __init__(self):
        self.regime_classifier = MarketRegimeClassifier()
        self.regime_models = {}
        self.base_models = {
            'lgb': lgb.LGBMRegressor(random_state=42),
            'rf': RandomForestRegressor(random_state=42),
            'svr': SVR(kernel='rbf'),
            'nn': MLPRegressor(random_state=42),
            'quantile': QuantileRegressor(quantile=0.5)
        }
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit two-stage model"""
        # First stage: Regime prediction
        regimes = self.regime_classifier.fit_predict(X)
        
        # Second stage: Conditional models
        for regime in range(self.regime_classifier.n_regimes):
            mask = regimes == regime
            if mask.sum() > 0:
                self.regime_models[regime] = {
                    name: clone(model).fit(X[mask], y[mask])
                    for name, model in self.base_models.items()
                }
                
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using two-stage approach"""
        regimes = self.regime_classifier.hmm.predict(
            self.regime_classifier.scaler.transform(X)
        )
        
        predictions = np.zeros(len(X))
        weights = self._calculate_regime_weights(X)
        
        for regime in range(self.regime_classifier.n_regimes):
            mask = regimes == regime
            if mask.sum() > 0:
                regime_preds = np.column_stack([
                    model.predict(X[mask])
                    for model in self.regime_models[regime].values()
                ])
                predictions[mask] = np.average(
                    regime_preds, axis=1, weights=weights[regime]
                )
                
        return predictions
    
    def _calculate_regime_weights(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate regime-based model weights"""
        # Simple equal weighting for now
        return np.ones(len(self.base_models)) / len(self.base_models)

class AdvancedFeatureSelector:
    """Advanced feature selection with temporal stability analysis"""
    
    def __init__(self):
        self.selected_features = None
        self.importance_scores = None
        self.stability_scores = None
        
    def calculate_iv(self, X: pd.DataFrame, y: pd.Series, bins=10) -> pd.Series:
        """Calculate Information Value for features"""
        iv_scores = {}
        
        for col in X.columns:
            try:
                # Bin the feature
                X_binned = pd.qcut(X[col], bins, duplicates='drop')
                
                # Calculate WOE and IV
                grouped = pd.DataFrame({
                    'target': y,
                    'bin': X_binned
                }).groupby('bin')
                
                woe_df = pd.DataFrame()
                woe_df['good'] = grouped['target'].apply(lambda x: (x >= x.mean()).sum())
                woe_df['bad'] = grouped['target'].apply(lambda x: (x < x.mean()).sum())
                
                woe_df['good_pct'] = woe_df['good'] / woe_df['good'].sum()
                woe_df['bad_pct'] = woe_df['bad'] / woe_df['bad'].sum()
                
                woe_df['woe'] = np.log(woe_df['good_pct'] / woe_df['bad_pct'])
                woe_df['iv'] = (woe_df['good_pct'] - woe_df['bad_pct']) * woe_df['woe']
                
                iv_scores[col] = woe_df['iv'].sum()
            except:
                iv_scores[col] = 0
                
        return pd.Series(iv_scores)
        
    def calculate_temporal_stability(
        self, 
        X: pd.DataFrame,
        window_size: int = 252
    ) -> pd.Series:
        """Calculate temporal stability of features"""
        stability_scores = {}
        
        for col in X.columns:
            # Calculate rolling statistics
            roll_mean = X[col].rolling(window_size).mean()
            roll_std = X[col].rolling(window_size).std()
            
            # Calculate coefficient of variation of these statistics
            cv_mean = roll_mean.std() / roll_mean.mean()
            cv_std = roll_std.std() / roll_std.mean()
            
            # Combine into single stability score
            stability_scores[col] = 1 / (1 + cv_mean + cv_std)
            
        return pd.Series(stability_scores)
        
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 20,
        stability_weight: float = 0.3
    ) -> List[str]:
        """Select features based on importance and stability"""
        # Calculate feature importance scores
        mi_scores = mutual_info_regression(X, y)
        iv_scores = self.calculate_iv(X, y)
        
        # Normalize scores
        mi_scores = pd.Series(mi_scores, index=X.columns)
        mi_scores = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())
        iv_scores = (iv_scores - iv_scores.min()) / (iv_scores.max() - iv_scores.min())
        
        # Calculate temporal stability
        stability_scores = self.calculate_temporal_stability(X)
        
        # Combine scores
        importance_scores = 0.5 * mi_scores + 0.5 * iv_scores
        final_scores = (1 - stability_weight) * importance_scores + stability_weight * stability_scores
        
        # Select top features
        self.selected_features = final_scores.nlargest(n_features).index.tolist()
        self.importance_scores = importance_scores
        self.stability_scores = stability_scores
        
        return self.selected_features

class DomainFeatureGenerator:
    """Generate domain-specific features for credit spread prediction"""
    
    def __init__(self, windows=[20, 60, 120]):
        self.windows = windows
        
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate domain-specific features"""
        df = df.copy()
        
        # Term structure indicators
        df['term_spread'] = df['us_3m_10y']
        
        # Risk appetite indicators
        df['vix_regime'] = pd.qcut(df['vix'], q=3, labels=[0, 1, 2])  # Changed to numeric labels
        df['vix_ma'] = df['vix'].rolling(window=20).mean()
        df['vix_std'] = df['vix'].rolling(window=20).std()
        
        # Credit risk premium
        df['hy_ig_spread'] = df['us_hy_oas'] - df['us_ig_oas']
        df['cad_ig_spread'] = df['cad_oas'] - df['us_ig_oas']
        
        # Market stress index
        df['market_stress'] = (
            (df['vix'] > df['vix'].rolling(window=60).mean()) & 
            (df['hy_ig_spread'] > df['hy_ig_spread'].rolling(window=60).mean())
        ).astype(int)
        
        # Economic indicators
        df['growth_inflation_gap'] = df['us_growth_surprises'] - df['us_inflation_surprises']
        df['lei_momentum'] = df['us_lei_yoy'].diff()
        
        # Technical indicators for spreads
        for window in self.windows:
            # Momentum
            df[f'cad_oas_mom{window}'] = df['cad_oas'].pct_change(window)
            df[f'us_hy_oas_mom{window}'] = df['us_hy_oas'].pct_change(window)
            df[f'us_ig_oas_mom{window}'] = df['us_ig_oas'].pct_change(window)
            
            # Volatility
            df[f'cad_oas_vol{window}'] = df['cad_oas'].rolling(window=window).std()
            df[f'us_hy_oas_vol{window}'] = df['us_hy_oas'].rolling(window=window).std()
            df[f'us_ig_oas_vol{window}'] = df['us_ig_oas'].rolling(window=window).std()
            
            # Z-scores
            df[f'cad_oas_zscore{window}'] = (
                (df['cad_oas'] - df['cad_oas'].rolling(window=window).mean()) / 
                df['cad_oas'].rolling(window=window).std()
            )
            
        # Cross-asset relationships
        df['tsx_momentum'] = df['tsx'].pct_change(20)
        df['tsx_volatility'] = df['tsx'].rolling(window=20).std()
        
        return df

class EnhancedCreditSpreadPredictor:
    """Enhanced credit spread predictor with regime awareness and robust validation"""
    
    def __init__(self):
        self.garch_generator = GARCHFeatureGenerator()
        self.cross_asset_generator = CrossAssetFeatureGenerator()
        self.domain_generator = DomainFeatureGenerator()
        self.feature_selector = AdvancedFeatureSelector()
        self.two_stage_model = TwoStageModel()
        self.scaler = RobustScaler()
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare enhanced feature set"""
        features = pd.DataFrame(index=df.index)
        
        # GARCH volatility features
        for col in ['cad_oas', 'us_ig_oas', 'us_hy_oas']:
            features[f'{col}_garch_vol'] = self.garch_generator.fit_predict(df[col])
            
        # Cross-asset features
        features = pd.concat([
            features, 
            self.cross_asset_generator.generate_features(df)
        ], axis=1)
        
        # Domain-specific features
        features = pd.concat([
            features,
            self.domain_generator.generate_features(df)
        ], axis=1)
        
        # Add original features
        features = pd.concat([features, df], axis=1)
        
        return features
        
    def select_features(self, X: pd.DataFrame, y: pd.Series, n_features: int = 20):
        """Select best features"""
        return self.feature_selector.select_features(X, y, n_features)
        
    def train_model(self, X: pd.DataFrame, y: pd.Series):
        """Train model with walk-forward optimization"""
        # Select features
        selected_features = self.select_features(X, y)
        X = X[selected_features]
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train two-stage model
            self.two_stage_model.fit(X_train_scaled, y_train)
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        # Use selected features
        X = X[self.feature_selector.selected_features]
        X_scaled = self.scaler.transform(X)
        return self.two_stage_model.predict(X_scaled)

class RobustCreditSpreadPredictor:
    """Enhanced credit spread predictor with robust feature engineering and model stacking"""
    
    def __init__(self, n_trials=100):
        """Initialize the predictor with default parameters."""
        self.n_trials = n_trials
        self.model_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': 42
        }
        # Initialize base models with default parameters
        self.base_models = {
            'lgb': lgb.LGBMRegressor(
                objective='regression',
                metric='rmse',
                boosting_type='gbdt',
                n_jobs=-1,
                random_state=42,
                max_depth=3,           # Reduced from 5
                min_child_samples=30,  # Increased from 20
                num_leaves=8,          # Reduced to prevent overfitting
                learning_rate=0.05,    # Reduced from 0.1
                n_estimators=100,      # Kept same
                colsample_bytree=0.8,  # Added to reduce overfitting
                subsample=0.8,         # Added to reduce overfitting
                reg_alpha=0.1,         # Added L1 regularization
                reg_lambda=0.1         # Added L2 regularization
            ),
            'rf': RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                max_depth=5,           # Added max_depth
                min_samples_split=5,   # Added min_samples_split
                min_samples_leaf=3     # Added min_samples_leaf
            ),
            'elastic': ElasticNet(
                random_state=42,
                max_iter=5000,         # Increased from 2000
                alpha=0.1,             # Added alpha
                l1_ratio=0.5,          # Added l1_ratio
                tol=1e-4               # Added tolerance
            )
        }
        
        # Initialize meta-model with more conservative parameters
        self.meta_model = lgb.LGBMRegressor(
            objective='regression',
            boosting_type='gbdt',
            learning_rate=0.01,
            num_leaves=8,              # Reduced from 31
            max_depth=3,               # Reduced from 6
            min_child_samples=30,      # Increased from 20
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_estimators=500,          # Reduced from 1000
            random_state=42
        )
        
        self.scaler = RobustScaler()  # Changed to RobustScaler for better outlier handling
        self.outlier_detector = IsolationForest(contamination=0.05, random_state=42)  # Reduced contamination
        self.selected_features = None
        self.feature_importances = None
        self.hmm_model = GaussianHMM(n_components=3, random_state=42)
        self.kf = KalmanFilter(dim_x=2, dim_z=1)  # State: [position, velocity]
        self.random_state = 42
        self.kalman_estimators = {}
        self.enhanced_predictor = EnhancedCreditSpreadPredictor()
        self.label_encoders = {}
        self.feature_selector = SelectKBest(score_func=mutual_info_regression, k='all')
        self.trained_models = None
        self.is_trained = False
        
    def _create_base_models(self) -> Dict[str, Any]:
        """Create the base models for stacking"""
        # LightGBM with adjusted parameters to handle small datasets better
        lgb_model = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            boosting_type='gbdt',
            n_jobs=-1,
            random_state=42
        )
        
        # Random Forest model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        # ElasticNet model
        elastic_model = ElasticNet(
            random_state=42,
            max_iter=1000
        )
        
        return {
            'lgb': lgb_model,
            'rf': rf_model,
            'elastic': elastic_model
        }

    def _asymmetric_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Custom asymmetric loss function with balanced directional penalty.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Asymmetric loss value
        """
        try:
            # Convert inputs to numpy arrays
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            
            # Ensure same length
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            # Calculate directional components
            actual_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            
            # Pad the directions to match original length
            actual_direction = np.pad(actual_direction, (1, 0), mode='edge')
            pred_direction = np.pad(pred_direction, (1, 0), mode='edge')
            
            # Calculate directional penalty
            direction_penalty = np.where(
                actual_direction * pred_direction < 0,
                2.0,  # Higher penalty for wrong direction
                1.0   # Normal penalty for correct direction
            )
            
            # Calculate base error
            base_error = np.abs(y_true - y_pred)
            
            # Apply asymmetric penalty
            penalized_error = base_error * direction_penalty
            
            return float(np.mean(penalized_error))
            
        except Exception as e:
            logger.error(f"Error in asymmetric loss calculation: {str(e)}")
            logger.error(traceback.format_exc())
            return float('inf')  # Return infinity on error

    def _optimize_model(self, model_name, model, X, y):
        """Optimize model hyperparameters with enhanced search space and asymmetric loss."""
        def objective(trial):
            if model_name == 'lgb':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'num_leaves': trial.suggest_int('num_leaves', 8, 64),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                    'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 0.1, log=True),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 3, 20),
                    'random_state': 42,
                    'n_jobs': -1,
                    'force_col_wise': True,
                    'boost_from_average': True,
                    'feature_fraction_seed': 42,
                    'bagging_seed': 42,
                    'drop_seed': 42,
                    'data_random_seed': 42,
                    'extra_trees': True,
                    'path_smooth': trial.suggest_float('path_smooth', 0.0, 2.0),
                    'verbose': -1
                }
                
                # Create model with current parameters
                model = lgb.LGBMRegressor(**params)
                
                # Implement time-series cross-validation
                tscv = TimeSeriesSplit(n_splits=5)
                scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Add sample weights based on recency
                    sample_weights = np.linspace(0.5, 1.0, len(X_train))
                    
                    # Fit model with early stopping
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        eval_metric='rmse',
                        early_stopping_rounds=50,
                        sample_weight=sample_weights,
                        callbacks=[
                            lgb.callback.early_stopping(50),
                            lgb.callback.log_evaluation(period=0)  # Suppress output
                        ]
                    )
                    
                    # Predict and calculate custom loss
                    y_pred = model.predict(X_val)
                    loss = self._asymmetric_loss(y_val.values, y_pred)
                    scores.append(loss)
                
                return np.mean(scores)
            
            # Handle other model types similarly...
            return 0.0

        # Create and run study with enhanced sampler
        sampler = optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=10,
            multivariate=True
        )
        
        study = optuna.create_study(
            direction='minimize',
            sampler=sampler
        )
        
        # Run optimization with pruning
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=3600,  # 1 hour timeout
            catch=(Exception,),
            callbacks=[
                lambda study, trial: print(f"Trial {trial.number} finished with value: {trial.value}")
            ]
        )
        
        return study.best_params

    def prepare_features(self, df: pd.DataFrame, features: List[str]):
        """Prepare features with optimized feature engineering."""
        logger.info("Preparing features with advanced engineering...")
        logger.info("Starting feature preparation")
        
        try:
            # Copy dataframe to avoid modifying original
            df_copy = df.copy()
            
            # Handle infinite values
            df_copy = df_copy.replace([np.inf, -np.inf], np.nan)
            
            # Handle missing values with forward fill then backward fill
            df_copy = df_copy.fillna(method='ffill').fillna(method='bfill')
            
            # Remove outliers using IQR method
            Q1 = df_copy.quantile(0.25)
            Q3 = df_copy.quantile(0.75)
            IQR = Q3 - Q1
            df_copy = df_copy[~((df_copy < (Q1 - 3 * IQR)) | (df_copy > (Q3 + 3 * IQR))).any(axis=1)]
            
            # Scale features using RobustScaler
            for col in features:
                if col != 'target':
                    scaler = RobustScaler()
                    df_copy[col] = scaler.fit_transform(df_copy[[col]])
            
            # Split into X and y
            X = df_copy[features]
            y = df_copy['target']
            
            logger.info(f"Data shape after cleaning: {X.shape}")
            logger.info(f"Final feature set shape: {X.shape}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error in prepare_features: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def optimize_models(self, X: pd.DataFrame, y: pd.Series):
        """Optimize all models using Optuna"""
        logger.info("Starting model optimization")
        
        # Optimize LightGBM
        lgb_params = self._optimize_model('lgb', self.base_models['lgb'], X, y)
        
        # Update base models with optimized parameters
        self.base_models['lgb'] = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            boosting_type='gbdt',
            n_jobs=-1,
            random_state=42
        )
        
        # Optimize Random Forest
        rf_params = self._optimize_model('rf', self.base_models['rf'], X, y)
        
        # Update Random Forest parameters
        self.base_models['rf'] = RandomForestRegressor(**rf_params, random_state=42)
        
        # Optimize ElasticNet
        elastic_params = self._optimize_model('elastic', self.base_models['elastic'], X, y)
        
        # Update ElasticNet parameters
        self.base_models['elastic'] = ElasticNet(**elastic_params, random_state=42)
        
        logger.info("Model optimization completed")

    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, Dict[str, float]]:
        """Train models with enhanced stacking and asymmetric loss."""
        try:
            # Create TimeSeriesSplit with smaller number of splits
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Initialize storage for predictions
            meta_features = np.zeros((X.shape[0], len(self.base_models)))
            
            # Train and predict with each base model
            for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                logger.info(f"Training fold {i+1}/5")
                
                for j, (name, model) in enumerate(self.base_models.items()):
                    logger.info(f"Training {name} model...")
                    
                    # Clone model to ensure fresh instance
                    model_clone = clone(model)
                    
                    # Add early stopping for LightGBM
                    if name == 'lgb':
                        eval_set = [(X_val, y_val)]
                        model_clone.fit(
                            X_train, y_train,
                            eval_set=eval_set,
                            callbacks=[
                                lgb.early_stopping(stopping_rounds=50),
                                lgb.log_evaluation(period=0)
                            ]
                        )
                    else:
                        model_clone.fit(X_train, y_train)
                    
                    # Make predictions
                    pred = model_clone.predict(X_val)
                    meta_features[val_idx, j] = pred
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_val, pred)
                    mae = mean_absolute_error(y_val, pred)
                    r2 = r2_score(y_val, pred)
                    
                    logger.info(f"{name} Fold {i+1} Metrics:")
                    logger.info(f"MSE: {mse:.4f}")
                    logger.info(f"MAE: {mae:.4f}")
                    logger.info(f"R2: {r2:.4f}")
            
            # Train final models on full dataset
            logger.info("Training final models on full dataset...")
            self.trained_models = {}
            for name, model in self.base_models.items():
                self.trained_models[name] = clone(model)
                if name == 'lgb':
                    eval_set = [(X, y)]
                    self.trained_models[name].fit(
                        X, y,
                        eval_set=eval_set,
                        callbacks=[lgb.log_evaluation(period=0)]
                    )
                else:
                    self.trained_models[name].fit(X, y)
            
            # Train meta-model
            logger.info("Training meta-model...")
            self.meta_model.fit(meta_features, y)
            
            self.is_trained = True
            
            # Make predictions using the trained models
            base_predictions = np.zeros((len(X), len(self.trained_models)))
            for i, model in enumerate(self.trained_models.values()):
                base_predictions[:, i] = model.predict(X)
            
            final_pred = self.meta_model.predict(base_predictions)
            
            # Calculate and log final performance metrics
            metrics = self.calculate_metrics(y, final_pred)
            
            logger.info("Final Model Performance:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
            return final_pred, metrics
            
        except Exception as e:
            logger.error(f"Error in train_models: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained ensemble"""
        try:
            if not hasattr(self, 'is_trained') or not self.is_trained:
                raise ValueError("Models have not been trained. Call train_models first.")

            # Generate base model predictions
            base_predictions = np.zeros((len(X), len(self.trained_models)))
            
            for i, (name, model) in enumerate(self.trained_models.items()):
                if not hasattr(model, 'predict'):
                    raise ValueError(f"Model {name} is not properly initialized")
                base_predictions[:, i] = model.predict(X)
            
            # Generate meta-model predictions
            if not hasattr(self.meta_model, 'predict'):
                raise ValueError("Meta-model is not properly initialized")
            final_predictions = self.meta_model.predict(base_predictions)
            
            return final_predictions
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        try:
            # Convert inputs to numpy arrays and ensure same length
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
                'explained_variance': explained_variance_score(y_true, y_pred),
                'max_error': max_error(y_true, y_pred),
                'custom_loss': self._asymmetric_loss(y_true, y_pred)
            }
            
            # Calculate directional accuracy
            true_diff = np.diff(y_true)
            pred_diff = np.diff(y_pred)
            direction_correct = np.mean(np.sign(true_diff) == np.sign(pred_diff))
            metrics['directional_accuracy'] = direction_correct
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in metric calculation: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'mse': float('inf'),
                'rmse': float('inf'),
                'mae': float('inf'),
                'r2': float('-inf'),
                'explained_variance': float('-inf'),
                'max_error': float('inf'),
                'custom_loss': float('inf'),
                'directional_accuracy': 0.0
            }

    def analyze_feature_importances(self, X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Analyze and aggregate feature importances from all models"""
        logger.info("Analyzing feature importances")
        importances = {}
        
        if not self.trained_models:
            logger.warning("Models not trained yet. Please train models before analyzing feature importances.")
            return {}

        # Get feature importances from each model
        for name, model in self.trained_models.items():
            try:
                if name == 'lgb':
                    # LightGBM feature importance
                    importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': model.feature_importances_
                    })
                elif name == 'rf':
                    # Random Forest feature importance
                    importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': model.feature_importances_
                    })
                elif name == 'elastic':
                    # ElasticNet coefficients as importance
                    importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': np.abs(model.coef_)  # Use absolute values
                    })
                
                # Sort by importance and normalize
                importance = importance.sort_values('importance', ascending=False)
                importance['importance_normalized'] = importance['importance'] / importance['importance'].sum()
                importances[name] = importance
                
                # Log top 10 features for each model
                logger.info(f"\nTop 10 important features for {name}:")
                for idx, row in importance.head(10).iterrows():
                    logger.info(f"{row['feature']}: {row['importance_normalized']:.4f}")
            except (AttributeError, NotFittedError) as e:
                logger.warning(f"Could not get feature importances for {name} model: {str(e)}")
                continue

        return importances

    def analyze_feature_interactions(self, X: pd.DataFrame, y: pd.Series, top_n: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyze feature interactions using a tree-based model and mutual information.
        
        Args:
            X: Input features DataFrame
            y: Target variable
            top_n: Number of top features to analyze interactions for
            
        Returns:
            Tuple of (feature importances DataFrame, interaction scores DataFrame)
        """
        logger.info("\nAnalyzing feature interactions...")
        
        # First get feature importances
        importances = self.analyze_feature_importances(X)
        if not importances:
            logger.warning("Could not analyze feature interactions: No feature importances available")
            return pd.DataFrame(), pd.DataFrame()
            
        # Get top features across all models
        all_top_features = set()
        for model_importances in importances.values():
            top_model_features = model_importances.nlargest(top_n, 'importance')['feature'].tolist()
            all_top_features.update(top_model_features)
            
        # Convert to list and limit to top_n
        top_features = list(all_top_features)[:top_n]
        
        # Create interaction matrix
        n_features = len(top_features)
        interaction_matrix = np.zeros((n_features, n_features))
        
        # Calculate pairwise mutual information
        for i, feat1 in enumerate(top_features):
            for j, feat2 in enumerate(top_features):
                if i != j:
                    # Calculate mutual information between features
                    mi_score = mutual_info_regression(
                        X[[feat1]], X[feat2].values.reshape(-1, 1)
                    )[0]
                    interaction_matrix[i, j] = mi_score
        
        # Create DataFrame for interaction scores
        interaction_df = pd.DataFrame(
            interaction_matrix,
            index=top_features,
            columns=top_features
        )
        
        # Log top interactions
        logger.info("\nTop feature interactions:")
        n_interactions = min(5, len(top_features) * (len(top_features) - 1) // 2)
        interactions = []
        for i in range(len(top_features)):
            for j in range(i + 1, len(top_features)):
                interactions.append({
                    'feature1': top_features[i],
                    'feature2': top_features[j],
                    'score': interaction_matrix[i, j]
                })
        
        interaction_results = pd.DataFrame(interactions)
        interaction_results = interaction_results.sort_values('score', ascending=False)
        
        for _, row in interaction_results.head(n_interactions).iterrows():
            logger.info(f"{row['feature1']} - {row['feature2']}: {row['score']:.4f}")
        
        return importances.get('lgb', pd.DataFrame()), interaction_df

    def test_interaction_significance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_n: int = 10,
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        Test statistical significance of feature interactions using 
        permutation tests and multiple hypothesis testing correction.
        
        Args:
            X: Input features DataFrame
            y: Target variable
            top_n: Number of top features to analyze
            alpha: Significance level for hypothesis tests
            
        Returns:
            DataFrame with interaction scores and p-values
        """
        logger.info("\nTesting feature interaction significance...")
        
        # Get base interaction scores
        _, interaction_df = self.analyze_feature_interactions(X, y)
        significant_interactions = interaction_df.head(top_n)
        
        # Perform permutation tests
        n_permutations = 1000
        p_values = []
        
        for _, row in significant_interactions.iterrows():
            feat1, feat2 = row['feature1'], row['feature2']
            base_score = row['score']
            
            # Calculate permutation distribution
            perm_scores = []
            for _ in range(n_permutations):
                # Permute one feature
                X_perm = X.copy()
                X_perm[feat2] = np.random.permutation(X_perm[feat2].values)
                
                # Calculate interaction score
                interaction = X_perm[feat1] * X_perm[feat2]
                mi_score = mutual_info_regression(
                    interaction.values.reshape(-1, 1),
                    y,
                    random_state=42
                )[0]
                
                perm_scores.append(mi_score)
            
            # Calculate p-value
            p_value = np.mean(np.array(perm_scores) >= base_score)
            p_values.append(p_value)
        
        # Correct for multiple testing
        reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        
        # Create results DataFrame
        interaction_df['p_value'] = p_values
        interaction_df['adjusted_p_value'] = pvals_corrected
        interaction_df['significant'] = reject
        
        # Log significant interactions
        significant_interactions = interaction_df[interaction_df['significant']]
        logger.info(f"\nFound {len(significant_interactions)} significant interactions "
                   f"(adjusted p < {alpha}):")
        
        for _, row in significant_interactions.iterrows():
            logger.info(
                f"{row['feature1']} × {row['feature2']}: "
                f"Score={row['score']:.4f}, "
                f"Adjusted p={row['adjusted_p_value']:.4f}"
            )
        
        return interaction_df

    def select_features_by_importance(self, X: pd.DataFrame, importance_threshold: float = 0.01) -> List[str]:
        """
        Select features based on their aggregate importance scores.
        
        Args:
            X: Input features DataFrame
            importance_threshold: Minimum importance score for feature selection
            
        Returns:
            List of selected feature names
        """
        importances = self.analyze_feature_importances(X)
        
        # Calculate aggregate importance across all models
        all_importances = pd.DataFrame()
        for name, imp in importances.items():
            if all_importances.empty:
                all_importances = imp[['feature', 'importance_normalized']].copy()
                all_importances.columns = ['feature', name]
            else:
                all_importances[name] = all_importances['feature'].map(
                    imp.set_index('feature')['importance_normalized']
                )
        
        all_importances['mean_importance'] = all_importances.iloc[:, 1:].mean(axis=1)
        selected_features = all_importances[all_importances['mean_importance'] > importance_threshold]['feature'].tolist()
        
        logger.info(f"Selected {len(selected_features)} features with importance > {importance_threshold}")
        return selected_features

    def select_stable_features(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            importance_threshold: float = 0.003,  # Further lowered
            cv_threshold: float = 0.95,  # Increased to be more lenient
            min_features: int = 20  # Increased minimum features
        ):
        """Select features based on both importance and stability metrics."""
        stability_results = self.analyze_feature_stability(X, y)
        feature_stats = stability_results['feature_stats']
        
        # Select features that meet both criteria
        selected_features = [
            feature for feature, stats in feature_stats.items()
            if (stats['mean'] >= importance_threshold and stats['cv'] <= cv_threshold)
        ]
        
        # Ensure minimum number of features
        if len(selected_features) < min_features:
            # Sort by importance/CV ratio and take top min_features
            feature_scores = {
                feature: stats['mean'] / (stats['cv'] + 1e-6)
                for feature, stats in feature_stats.items()
            }
            selected_features = sorted(
                feature_scores.keys(),
                key=lambda x: feature_scores[x],
                reverse=True
            )[:min_features]
        
        return selected_features

    def analyze_feature_stability(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict:
        """
        Analyze feature importance stability across different time periods.
        
        Args:
            X: Input features DataFrame
            y: Target variable
            n_splits: Number of time series splits for stability analysis
            
        Returns:
            Dictionary with feature importance statistics across splits
        """
        feature_importances = []
        cv = TimeSeriesSplit(n_splits=n_splits)
        
        # Calculate feature importance for each fold
        for train_idx, _ in cv.split(X):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            
            model = LGBMRegressor(**self.model_params)
            model.fit(X_train, y_train)
            
            importance = pd.Series(model.feature_importances_, index=X.columns)
            feature_importances.append(importance)
        
        # Calculate mean and CV of feature importance
        importance_df = pd.DataFrame(feature_importances)
        mean_importance = importance_df.mean()
        cv_importance = importance_df.std() / importance_df.mean()
        
        # Create feature stats dictionary
        feature_stats = {}
        for feature in X.columns:
            feature_stats[feature] = {
                'mean': mean_importance[feature],
                'cv': cv_importance[feature]
            }
        
        # Select stable features based on criteria
        stable_features = [
            feature for feature, stats in feature_stats.items()
            if stats['mean'] > 0.01 and stats['cv'] < 0.75
        ]
        
        return {
            'feature_stats': feature_stats,
            'selected_features': stable_features,
            'mean_cv': cv_importance.mean(),
            'mean_importance': mean_importance.mean(),
            'stability_score': len(stable_features) / len(X.columns)
        }

    def plot_feature_importances(self, importances: Dict[str, pd.DataFrame], save_path: str = 'feature_importances.png'):
        """Create and save feature importance plots"""
        plt.figure(figsize=(12, 8))
        
        # Create a subplot for each model
        n_models = len(importances)
        fig, axes = plt.subplots(n_models + 1, 1, figsize=(12, 4 * (n_models + 1)))
        
        # Plot individual model importances
        for i, (name, importance) in enumerate(importances.items()):
            top_features = importance.head(10)
            sns.barplot(x='importance_normalized', y='feature', data=top_features, ax=axes[i])
            axes[i].set_title(f'Top 10 Feature Importances - {name}')
            axes[i].set_xlabel('Normalized Importance')
            
        # Plot aggregate importance
        all_importances = pd.DataFrame()
        for name, imp in importances.items():
            if all_importances.empty:
                all_importances = imp[['feature', 'importance_normalized']].copy()
                all_importances.columns = ['feature', name]
            else:
                all_importances[name] = all_importances['feature'].map(
                    imp.set_index('feature')['importance_normalized']
                )
        
        all_importances['mean_importance'] = all_importances.iloc[:, 1:].mean(axis=1)
        all_importances = all_importances.sort_values('mean_importance', ascending=False).head(10)
        
        sns.barplot(x='mean_importance', y='feature', data=all_importances, ax=axes[-1])
        axes[-1].set_title('Top 10 Feature Importances - Aggregate')
        axes[-1].set_xlabel('Mean Normalized Importance')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Feature importance plots saved to {save_path}")

def main():
    """Main execution function"""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        logger.info("Starting enhanced credit spread prediction model training")
        
        # Load and prepare data
        predictor = RobustCreditSpreadPredictor()
        logger.info("Preparing features with advanced engineering...")
        
        # Load data and prepare features
        logger.info("Starting feature preparation")
        
        # Load data from the correct path
        data = pd.read_csv('pulling_data/backtest_data.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        
        # Create target variable (1 period forward)
        data['target'] = data['cad_oas'].shift(-1)
        data = data.dropna()  # Drop rows with NaN values
        
        # Get features
        features = [col for col in data.columns if col not in ['target']]
        
        # Prepare features
        X = data[features]
        y = data['target']
        
        logger.info(f"Data shape after cleaning: {X.shape}")
        logger.info(f"Final feature set shape: {X.shape}")
        
        # Feature selection and stability analysis
        logger.info("\nPerforming feature selection and stability analysis...")
        logger.info("\nSelecting stable features...")
        
        selected_features = predictor.select_stable_features(X, y)
        
        # Train model with walk-forward optimization
        logger.info("\nTraining model with walk-forward optimization...")
        
        # Train models
        predictor.train_models(X, y)
        
        # Calculate and log performance metrics
        metrics = predictor.calculate_metrics(y, predictor.predict(X))
        
        logger.info("\nFinal Model Performance:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Analyze feature interactions
        logger.info("\nAnalyzing feature interactions...")
        importances, interaction_df = predictor.analyze_feature_interactions(X, y)
        
        # Log feature interaction results
        logger.info("\nTop Feature Interactions:")
        for i in range(min(5, len(interaction_df.index))):
            for j in range(i + 1, len(interaction_df.columns)):
                feat1 = interaction_df.index[i]
                feat2 = interaction_df.columns[j]
                score = interaction_df.iloc[i, j]
                if score > 0:
                    logger.info(f"{feat1} × {feat2}: Interaction Strength = {score:.4f}")
        
        # Evaluate model performance
        logger.info("\nEvaluating model performance...")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()
