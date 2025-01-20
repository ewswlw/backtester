import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.svm import SVR
from sklearn.linear_model import LassoCV, ElasticNet
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, max_error
from sklearn.base import clone
import lightgbm as lgb
import logging
import traceback
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RobustCreditSpreadPredictor:
    """Enhanced credit spread predictor with robust feature engineering and model stacking"""
    
    def __init__(self):
        """Initialize the predictor with enhanced components"""
        logger.info("Initializing RobustCreditSpreadPredictor")
        self.scaler = StandardScaler()
        self.outlier_detector = IsolationForest(contamination=0.1, random_state=42)
        self.selected_features = None
        self.feature_importances = None
        self.base_models = {}
        self.meta_model = None
        self.create_stacked_model()
        
    def load_data(self, file_path):
        """Load and preprocess data with enhanced logging"""
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Convert date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        logger.info(f"Data timespan: {df.index.min()} to {df.index.max()}")
        
        # Drop index columns
        index_cols = [col for col in df.columns if 'er_index' in col]
        df = df.drop(columns=index_cols)
        logger.info(f"Dropped index columns: {index_cols}")
        logger.info(f"Data shape: {df.shape}")
        
        # Create target variable (1 period forward)
        df['target'] = df['cad_oas'].shift(-1)
        
        # Get features
        features = [col for col in df.columns if col not in ['target']]
        logger.info(f"Features used: {features}")
        
        return df, features

    def _add_lagged_features(self, X, df):
        """Add lagged features for key variables"""
        key_vars = ['cad_oas', 'us_hy_oas', 'us_ig_oas']
        lags = [1, 3, 6, 12]
        
        for var in key_vars:
            if var in df.columns:
                for lag in lags:
                    X[f'{var}_lag{lag}'] = df[var].shift(lag)
        
        logger.info(f"Added lagged features for {key_vars}")
        return X

    def _add_moving_averages(self, X, df):
        """Add moving averages for key variables"""
        key_vars = ['cad_oas', 'us_hy_oas', 'us_ig_oas']
        windows = [3, 6, 12]
        
        for var in key_vars:
            if var in df.columns:
                for window in windows:
                    X[f'{var}_ma{window}'] = df[var].rolling(window=window).mean()
                    X[f'{var}_std{window}'] = df[var].rolling(window=window).std()
        
        logger.info(f"Added moving averages for {key_vars}")
        return X

    def _add_momentum_features(self, X, df):
        """Add momentum indicators"""
        key_vars = ['cad_oas', 'us_hy_oas', 'us_ig_oas']
        windows = [1, 3, 6, 12]
        
        for var in key_vars:
            if var in df.columns:
                for window in windows:
                    X[f'{var}_mom{window}'] = df[var].pct_change(window)
        
        logger.info("Added momentum features")
        return X

    def _add_spread_features(self, X, df):
        """Add spread features between different OAS"""
        if all(col in df.columns for col in ['us_hy_oas', 'us_ig_oas']):
            X['hy_ig_spread'] = df['us_hy_oas'] - df['us_ig_oas']
        
        logger.info("Added spread features")
        return X

    def _add_technical_indicators(self, X, df):
        """Add technical indicators"""
        key_vars = ['cad_oas', 'us_hy_oas', 'us_ig_oas']
        
        for var in key_vars:
            if var in df.columns:
                # RSI
                delta = df[var].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                X[f'{var}_rsi'] = 100 - (100 / (1 + rs))
                
                # Bollinger Bands
                ma20 = df[var].rolling(window=20).mean()
                std20 = df[var].rolling(window=20).std()
                X[f'{var}_bb_upper'] = ma20 + (std20 * 2)
                X[f'{var}_bb_lower'] = ma20 - (std20 * 2)
        
        logger.info("Added technical indicators")
        return X

    def handle_outliers(self, X):
        """Handle outliers using IQR method"""
        try:
            # Calculate IQR for each column
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Create a mask for non-outlier values
            mask = ~((X < lower_bound) | (X > upper_bound)).any(axis=1)
            
            # Apply mask and log the number of outliers removed
            X_clean = X[mask].copy()
            n_outliers = len(X) - len(X_clean)
            logger.info(f"Removed {n_outliers} outliers")
            
            return X_clean
            
        except Exception as e:
            logger.error(f"Error in handle_outliers: {str(e)}")
            logger.error(traceback.format_exc())
            return X

    def detect_outliers(self, X):
        """Detect and handle outliers using Isolation Forest"""
        outlier_labels = self.outlier_detector.fit_predict(X)
        return outlier_labels == 1
    
    def add_economic_regime_features(self, X, df):
        """Add economic regime features with proper NaN handling"""
        logger.info("Added economic regime features")
        
        try:
            # Define features for regime detection
            vol_features = ['vix', 'us_3m_10y']
            growth_features = ['us_growth_surprises', 'us_lei_yoy']
            
            # Handle NaN values before clustering
            X_regime = X.copy()
            X_regime = X_regime.fillna(method='ffill').fillna(method='bfill')
            
            # Volatility regime
            kmeans = KMeans(n_clusters=3, random_state=42)
            if all(feat in X_regime.columns for feat in vol_features):
                vol_data = X_regime[vol_features]
                X['vol_regime_cluster'] = kmeans.fit_predict(vol_data)
            
            # Growth regime
            if all(feat in X_regime.columns for feat in growth_features):
                growth_data = X_regime[growth_features]
                X['growth_regime_cluster'] = kmeans.fit_predict(growth_data)
            
            # Add interaction between regimes
            if 'vol_regime_cluster' in X.columns and 'growth_regime_cluster' in X.columns:
                X['combined_regime'] = X['vol_regime_cluster'] * 3 + X['growth_regime_cluster']
            
            return X
            
        except Exception as e:
            logger.error(f"Error in add_economic_regime_features: {str(e)}")
            logger.error(traceback.format_exc())
            return X  # Return original DataFrame if error occurs
    
    def add_advanced_nonlinear_features(self, X):
        """Add advanced non-linear transformations"""
        key_features = [col for col in ['us_hy_oas', 'vix', 'us_3m_10y', 'us_growth_surprises'] if col in X.columns]
        
        for feat in key_features:
            # Polynomial features
            X[f'{feat}_cubic'] = X[feat] ** 3
            
            # Exponential features
            X[f'{feat}_exp'] = np.exp(X[feat] / X[feat].std())
            
            # Log-transform for positive features
            if (X[feat] > 0).all():
                X[f'{feat}_log'] = np.log1p(X[feat])
            
            # Sigmoid transformation
            X[f'{feat}_sigmoid'] = 1 / (1 + np.exp(-X[feat]))
        
        logger.info("Added non-linear features")
        return X
    
    def add_interaction_features(self, X):
        """Add interaction features between economic indicators"""
        eco_features = [col for col in ['us_growth_surprises', 'us_inflation_surprises', 'us_lei_yoy'] if col in X.columns]
        market_features = [col for col in ['vix', 'us_3m_10y'] if col in X.columns]
        
        # Interactions between economic indicators
        for i, feat1 in enumerate(eco_features):
            for feat2 in eco_features[i+1:]:
                X[f'{feat1}_{feat2}_interact'] = X[feat1] * X[feat2]
        
        # Interactions with market features
        for eco_feat in eco_features:
            for market_feat in market_features:
                X[f'{eco_feat}_{market_feat}_interact'] = X[eco_feat] * X[market_feat]
        
        logger.info("Added interaction features")
        return X
    
    def calculate_directional_accuracy(self, y_true, y_pred):
        """Calculate directional accuracy with improved methodology"""
        try:
            # Convert to pandas Series if needed
            if not isinstance(y_true, pd.Series):
                y_true = pd.Series(y_true)
            if not isinstance(y_pred, pd.Series):
                y_pred = pd.Series(y_pred, index=y_true.index)
                
            # Calculate percent changes
            true_changes = y_true.pct_change()
            pred_changes = y_pred.pct_change()
            
            # Handle edge cases
            true_changes = true_changes.replace([np.inf, -np.inf], np.nan)
            pred_changes = pred_changes.replace([np.inf, -np.inf], np.nan)
            
            # Get common valid indices
            valid_idx = true_changes.notna() & pred_changes.notna()
            
            # Calculate directional accuracy
            correct_directions = (np.sign(true_changes[valid_idx]) == np.sign(pred_changes[valid_idx])).sum()
            total_valid = valid_idx.sum()
            
            return correct_directions / total_valid if total_valid > 0 else 0
            
        except Exception as e:
            logger.warning(f"Error calculating directional accuracy: {str(e)}")
            return 0

    def create_stacked_model(self):
        """Create a stacked model with optimized base models"""
        # Base models with tuned parameters
        self.base_models = {
            'rf': RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                min_samples_split=5,
                random_state=42
            ),
            'elastic': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=42
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                random_state=42
            )
        }
        
        # Meta-model with adjusted parameters
        self.meta_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    def select_stable_features(self, X, y, threshold=0.5):
        """Select features that are stable across multiple iterations"""
        try:
            logger.info("Starting feature selection")
            
            # Remove highly correlated features first
            correlation_matrix = X.corr().abs()
            upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            X = X.drop(columns=to_drop)
            logger.info(f"Removed {len(to_drop)} highly correlated features")
            
            # Initialize feature importance collector
            feature_importances = []
            n_iterations = 5
            
            # Collect feature importance scores across multiple iterations
            for i in range(n_iterations):
                # Create and train a LightGBM model
                model = lgb.LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42 + i
                )
                model.fit(X, y)
                
                # Get feature importance scores
                importance_scores = model.feature_importances_
                feature_importances.append(importance_scores)
            
            # Calculate mean and std of feature importance scores
            mean_importance = np.mean(feature_importances, axis=0)
            std_importance = np.std(feature_importances, axis=0)
            
            # Create feature importance DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance_mean': mean_importance,
                'importance_std': std_importance
            })
            
            # Select stable features based on importance mean and stability
            stability_score = feature_importance_df['importance_mean'] / (feature_importance_df['importance_std'] + 1e-6)
            selected_features = feature_importance_df[stability_score > threshold]['feature'].tolist()
            
            logger.info(f"Selected {len(selected_features)} stable features")
            return selected_features
            
        except Exception as e:
            logger.error(f"Error in select_stable_features: {str(e)}")
            logger.error(traceback.format_exc())
            return X.columns.tolist()  # Return all features if selection fails

    def prepare_features(self, df, features):
        """Prepare features with comprehensive engineering and logging"""
        try:
            # 1. Handle missing values
            logger.info("Handling missing values")
            df = df.copy()
            df = df.fillna(method='ffill').fillna(method='bfill')
            logger.info(f"Data shape after initial cleaning: {df.shape}")
            
            # 2. Create feature matrix X and target y
            X = df[features].copy()
            y = df['cad_oas'].copy()
            
            # 3. Add basic features
            logger.info("Adding basic features")
            X = self._add_lagged_features(X, df)
            X = self._add_moving_averages(X, df)
            X = self._add_momentum_features(X, df)
            X = self._add_spread_features(X, df)
            X = self._add_technical_indicators(X, df)
            
            # 4. Add advanced features
            logger.info("Adding advanced features")
            X = self.add_economic_regime_features(X, df)
            X = self.add_advanced_nonlinear_features(X)
            X = self.add_interaction_features(X)
            
            logger.info(f"Data shape after cleaning: {X.shape}")
            
            # 5. Remove outliers
            X = self.handle_outliers(X)
            logger.info(f"Data shape after outlier removal: {X.shape}")
            
            # Align target with cleaned features
            y = y.loc[X.index]
            
            # 6. Feature selection
            logger.info("Performing feature selection")
            selected_features = self.select_stable_features(X, y)
            X = X[selected_features]  # This is now a DataFrame with selected features
            logger.info(f"Selected {len(X.columns)} stable features")
            
            # 7. Scale features
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            return X_scaled, y
            
        except Exception as e:
            logger.error(f"Error during feature preparation: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def train_models(self, X, y):
        """Train base models and meta-model with enhanced logging"""
        try:
            logger.info("Starting model training")
            
            # Handle NaN values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            # Time series cross-validation with larger validation size
            tscv = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = {name: [] for name in self.base_models.keys()}
            meta_features = np.zeros((X.shape[0], len(self.base_models)))
            
            # Train and evaluate each fold
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
                logger.info(f"Training fold {fold}")
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train and evaluate each base model
                for i, (name, model) in enumerate(self.base_models.items()):
                    model_clone = clone(model)
                    
                    # Special handling for LightGBM
                    if name == 'lgb':
                        eval_set = [(X_val, y_val)]
                        model_clone.fit(
                            X_train, y_train,
                            eval_set=eval_set,
                            eval_metric='rmse',
                            callbacks=[lgb.early_stopping(stopping_rounds=50)]
                        )
                    else:
                        model_clone.fit(X_train, y_train)
                    
                    # Make predictions for this fold
                    fold_pred = model_clone.predict(X_val)
                    meta_features[val_idx, i] = fold_pred
                    
                    # Calculate and log performance metrics
                    r2 = r2_score(y_val, fold_pred)
                    cv_scores[name].append(r2)
                    logger.info(f"Fold {fold} - {name} score (R2): {r2:.4f}")
            
            # Log average performance for each model
            for name, scores in cv_scores.items():
                avg_score = np.mean(scores)
                std_score = np.std(scores)
                logger.info(f"{name} - Average R2: {avg_score:.4f} (±{std_score:.4f})")
            
            # Train meta-model on the full meta-features
            self.meta_model.fit(meta_features, y)
            
            # Final training of base models
            for name, model in self.base_models.items():
                if name == 'lgb':
                    eval_set = [(X, y)]
                    model.fit(
                        X, y,
                        eval_set=eval_set,
                        eval_metric='rmse',
                        callbacks=[lgb.early_stopping(stopping_rounds=50)]
                    )
                else:
                    model.fit(X, y)
            
            # Make final predictions
            base_predictions = np.column_stack([
                model.predict(X) for model in self.base_models.values()
            ])
            predictions = self.meta_model.predict(base_predictions)
            
            # Calculate and return final metrics
            metrics = {}
            metrics['rmse'] = np.sqrt(mean_squared_error(y, predictions))
            metrics['mae'] = mean_absolute_error(y, predictions)
            metrics['r2'] = r2_score(y, predictions)
            metrics['explained_variance'] = explained_variance_score(y, predictions)
            metrics['max_error'] = max_error(y, predictions)
            metrics['mape'] = np.mean(np.abs((y - predictions) / y)) * 100
            metrics['directional_accuracy'] = self.calculate_directional_accuracy(y, predictions)
            
            logger.info("Final model performance metrics:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def predict(self, X):
        """Make predictions using the trained models."""
        # Fill NaN values with column means
        X = X.fillna(X.mean())
        
        # Generate predictions from base models
        meta_features = np.column_stack([
            model.predict(X) for model in self.base_models.values()
        ])
        return self.meta_model.predict(meta_features)

    def evaluate(self, X, y):
        """Evaluate model performance with detailed metrics"""
        metrics = {
            'rmse': 0,
            'mae': 0,
            'r2': 0,
            'explained_variance': 0,
            'max_error': 0,
            'mape': 0,
            'directional_accuracy': 0
        }
        
        # Make predictions
        logger.info("Making predictions")
        predictions = self.predict(X)
        
        # Calculate metrics
        metrics['rmse'] = np.sqrt(mean_squared_error(y, predictions))
        metrics['mae'] = mean_absolute_error(y, predictions)
        metrics['r2'] = r2_score(y, predictions)
        metrics['explained_variance'] = explained_variance_score(y, predictions)
        metrics['max_error'] = max_error(y, predictions)
        metrics['mape'] = np.mean(np.abs((y - predictions) / y)) * 100
        metrics['directional_accuracy'] = self.calculate_directional_accuracy(y, predictions)
        
        # Log all metrics
        logger.info("\nDetailed Performance Metrics:")
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"R²: {metrics['r2']:.4f}")
        logger.info(f"Explained Variance: {metrics['explained_variance']:.4f}")
        logger.info(f"Maximum Error: {metrics['max_error']:.4f}")
        logger.info(f"MAPE: {metrics['mape']:.4f}%")
        logger.info(f"Directional Accuracy: {metrics['directional_accuracy']:.4f}")
        
        return metrics

    def evaluate_model(self, X, y):
        """Evaluate model performance using cross-validation with comprehensive metrics"""
        metrics = {
            'rmse': [],
            'mae': [],
            'r2': [],
            'directional_accuracy': []
        }
        
        # Handle NaN values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        # Create KFold cross-validator
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train models
            logger.info(f"Training fold {fold}")
            
            # Train base models
            base_predictions = {}
            for name, model in self.base_models.items():
                if isinstance(model, lgb.LGBMRegressor):
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        eval_metric='rmse',
                        callbacks=[lgb.early_stopping(stopping_rounds=50)]
                    )
                else:
                    model.fit(X_train, y_train)
                base_predictions[name] = model.predict(X_val)
                logger.info(f"Fold {fold} - {name} score (R2): {r2_score(y_val, base_predictions[name]):.4f}")
            
            # Create meta-features
            meta_features = np.column_stack(list(base_predictions.values()))
            
            # Train meta-model
            self.meta_model.fit(meta_features, y_val)
            
            # Make final predictions
            y_pred = self.meta_model.predict(meta_features)
            
            # Calculate fold metrics
            fold_metrics = {
                'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
                'mae': mean_absolute_error(y_val, y_pred),
                'r2': r2_score(y_val, y_pred),
                'directional_accuracy': self.calculate_directional_accuracy(y_val, y_pred)
            }
            
            # Store metrics
            for metric, value in fold_metrics.items():
                metrics[metric].append(value)
            
            # Log fold metrics
            logger.info(f"\nFold {fold} Metrics:")
            logger.info(f"RMSE: {fold_metrics['rmse']:.4f}")
            logger.info(f"MAE: {fold_metrics['mae']:.4f}")
            logger.info(f"R²: {fold_metrics['r2']:.4f}")
            logger.info(f"Directional Accuracy: {fold_metrics['directional_accuracy']:.4f}")
        
        # Calculate average metrics and confidence intervals
        results = {}
        for metric in metrics:
            mean_val = np.mean(metrics[metric])
            ci = np.std(metrics[metric]) * 1.96 / np.sqrt(len(metrics[metric]))
            results[f'{metric}_mean'] = mean_val
            results[f'{metric}_ci'] = ci
        
        # Log cross-validation results
        logger.info("\nCross-validation Results (with 95% CI):")
        logger.info(f"RMSE: {results['rmse_mean']:.4f} (±{results['rmse_ci']:.4f})")
        logger.info(f"MAE: {results['mae_mean']:.4f} (±{results['mae_ci']:.4f})")
        logger.info(f"R²: {results['r2_mean']:.4f} (±{results['r2_ci']:.4f})")
        logger.info(f"Directional Accuracy: {results['directional_accuracy_mean']:.4f} (±{results['directional_accuracy_ci']:.4f})")
        
        return results

def load_data(file_path):
    """Load and preprocess data with enhanced logging"""
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    logger.info(f"Data timespan: {df.index.min()} to {df.index.max()}")
    
    # Drop index columns
    index_cols = [col for col in df.columns if 'er_index' in col]
    df = df.drop(columns=index_cols)
    logger.info(f"Dropped index columns: {index_cols}")
    logger.info(f"Data shape: {df.shape}")
    
    # Create target variable (1 period forward)
    df['target'] = df['cad_oas'].shift(-1)
    
    # Get features
    features = [col for col in df.columns if col not in ['target']]
    logger.info(f"Features used: {features}")
    
    return df, features

def get_features(df):
    """Get features from dataframe"""
    features = [col for col in df.columns if col not in ['target']]
    return features

def main():
    """Main execution function with enhanced logging"""
    try:
        logger.info("Starting credit spread prediction model training")
        logger.info("Initializing RobustCreditSpreadPredictor")
        predictor = RobustCreditSpreadPredictor()
        
        # Load and prepare data
        logger.info("Loading data from pulling_data/backtest_data.csv")
        df, features = load_data('pulling_data/backtest_data.csv')
        
        # Prepare features and train model
        X, y = predictor.prepare_features(df, features)
        
        # Train models and get metrics
        metrics = predictor.train_models(X, y)
        cv_metrics = predictor.evaluate_model(X, y)
        
        # Log full dataset performance
        logger.info("\nFull Dataset Performance:")
        test_metrics = predictor.evaluate(X, y)
        logger.info(f"RMSE: {test_metrics['rmse']:.4f}")
        logger.info(f"MAE: {test_metrics['mae']:.4f}")
        logger.info(f"R²: {test_metrics['r2']:.4f}")
        logger.info(f"Explained Variance: {test_metrics['explained_variance']:.4f}")
        logger.info(f"Maximum Error: {test_metrics['max_error']:.4f}")
        logger.info(f"MAPE: {test_metrics['mape']:.4f}%")
        logger.info(f"Directional Accuracy: {test_metrics['directional_accuracy']:.4f}")
        
        # Log cross-validation performance
        logger.info("\nCross-validation Performance:")
        logger.info(f"RMSE: {cv_metrics['rmse_mean']:.4f} (±{cv_metrics['rmse_ci']:.4f})")
        logger.info(f"MAE: {cv_metrics['mae_mean']:.4f} (±{cv_metrics['mae_ci']:.4f})")
        logger.info(f"R²: {cv_metrics['r2_mean']:.4f} (±{cv_metrics['r2_ci']:.4f})")
        logger.info(f"Directional Accuracy: {cv_metrics['directional_accuracy_mean']:.4f} (±{cv_metrics['directional_accuracy_ci']:.4f})")
        
        logger.info("Training completed successfully!")
        return test_metrics, cv_metrics
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
