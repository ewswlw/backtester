import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
import time

class MLCADOASPredictor:
    def __init__(self):
        self.models = {
            'ElasticNet': ElasticNet(
                alpha=0.001,
                l1_ratio=0.1,
                max_iter=5000,
                random_state=42
            )
        }
        self.best_model = None
        self.best_model_name = None
        self.best_score = float('-inf')
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.selected_features = None
        
    def load_and_preprocess(self, data_path):
        """Load and preprocess the data"""
        print("Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Drop index columns
        columns_to_drop = ['cad_ig_er_index', 'us_hy_er_index', 'us_ig_er_index']
        df = df.drop(columns=columns_to_drop)
        print(f"Dropped index columns: {columns_to_drop}")
        
        # Create target (forward OAS)
        df['target'] = df['cad_oas'].shift(-1)
        
        # Remove target from features
        features = [col for col in df.columns if col not in ['target', 'cad_oas']]
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        print(f"Data shape: {df.shape}")
        print(f"Features used: {features}")
        
        return df, features
        
    def reduce_multicollinearity(self, X, threshold=0.95):
        """Remove highly correlated features"""
        print("Reducing multicollinearity...")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        print(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
        
        return [col for col in X.columns if col not in to_drop]
        
    def prepare_features(self, df, features):
        """Prepare features for modeling"""
        print("Preparing features...")
        
        # Select features and target
        X = df[features].copy()
        y = df['target'].copy()
        
        # 1. Lagged Features (Autoregressive Components)
        for lag in [1, 2, 3, 6]:
            X[f'cad_oas_lag{lag}'] = df['cad_oas'].shift(lag)
            X[f'us_hy_oas_lag{lag}'] = df['us_hy_oas'].shift(lag)
            X[f'us_ig_oas_lag{lag}'] = df['us_ig_oas'].shift(lag)
        
        # 2. Moving Averages and Volatility
        for window in [3, 6, 12]:
            # Moving averages
            X[f'cad_oas_ma{window}'] = df['cad_oas'].rolling(window=window).mean()
            X[f'us_hy_oas_ma{window}'] = df['us_hy_oas'].rolling(window=window).mean()
            
            # Volatility (standard deviation)
            X[f'cad_oas_std{window}'] = df['cad_oas'].rolling(window=window).std()
            X[f'us_hy_oas_std{window}'] = df['us_hy_oas'].rolling(window=window).std()
            
            # Exponential moving averages
            X[f'cad_oas_ema{window}'] = df['cad_oas'].ewm(span=window).mean()
            X[f'us_hy_oas_ema{window}'] = df['us_hy_oas'].ewm(span=window).mean()
        
        # 3. Momentum and Rate of Change
        for period in [1, 3, 6, 12]:
            # Momentum (percent change)
            X[f'cad_oas_mom{period}'] = df['cad_oas'].pct_change(period)
            X[f'us_hy_oas_mom{period}'] = df['us_hy_oas'].pct_change(period)
            
            # Rate of change
            X[f'cad_oas_roc{period}'] = (df['cad_oas'] - df['cad_oas'].shift(period)) / df['cad_oas'].shift(period) * 100
            X[f'us_hy_oas_roc{period}'] = (df['us_hy_oas'] - df['us_hy_oas'].shift(period)) / df['us_hy_oas'].shift(period) * 100
        
        # 4. Spread Features
        # Credit spread differentials
        X['hy_ig_spread'] = df['us_hy_oas'] - df['us_ig_oas']
        X['cad_hy_spread'] = df['cad_oas'] - df['us_hy_oas']
        X['cad_ig_spread'] = df['cad_oas'] - df['us_ig_oas']
        
        # Spread ratios
        X['hy_ig_ratio'] = df['us_hy_oas'] / df['us_ig_oas']
        X['cad_hy_ratio'] = df['cad_oas'] / df['us_hy_oas']
        
        # Moving averages of spreads
        for window in [3, 6, 12]:
            X[f'hy_ig_spread_ma{window}'] = X['hy_ig_spread'].rolling(window=window).mean()
            X[f'cad_hy_spread_ma{window}'] = X['cad_hy_spread'].rolling(window=window).mean()
        
        # 5. Technical Indicators
        # RSI-like indicator for credit spreads
        def calculate_rsi(series, periods=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        X['cad_oas_rsi'] = calculate_rsi(df['cad_oas'])
        X['us_hy_oas_rsi'] = calculate_rsi(df['us_hy_oas'])
        
        # Bollinger Bands
        for window in [20]:
            rolling_mean = df['cad_oas'].rolling(window=window).mean()
            rolling_std = df['cad_oas'].rolling(window=window).std()
            X[f'cad_oas_bb_upper{window}'] = rolling_mean + (rolling_std * 2)
            X[f'cad_oas_bb_lower{window}'] = rolling_mean - (rolling_std * 2)
            X[f'cad_oas_bb_width{window}'] = (X[f'cad_oas_bb_upper{window}'] - X[f'cad_oas_bb_lower{window}']) / rolling_mean
        
        # 6. Market Regime Features
        # Trend strength
        X['trend_strength'] = abs(X['cad_oas_ma12'] - X['cad_oas_ma3']) / X['cad_oas_std12']
        
        # Volatility regime
        X['vol_regime'] = X['cad_oas_std12'].rolling(window=3).mean() / X['cad_oas_std12'].rolling(window=12).mean()
        
        # Cross-market indicators
        X['market_stress'] = (df['vix'] * X['hy_ig_spread']) / 100
        X['yield_curve_momentum'] = df['us_3m_10y'].diff(3)
        
        # 7. Interaction Features
        X['vix_spread_interaction'] = df['vix'] * X['hy_ig_spread']
        X['curve_spread_interaction'] = df['us_3m_10y'] * X['hy_ig_spread']
        
        # Drop rows with NaN from the lag/rolling features
        X = X.dropna()
        y = y[X.index]
        
        # Remove low variance features
        var_thresh = VarianceThreshold(threshold=0.01)
        var_thresh.fit(X)
        mask = var_thresh.get_support()
        features = X.columns[mask].tolist()
        X = X[features]
        
        # Reduce multicollinearity
        self.selected_features = self.reduce_multicollinearity(X, threshold=0.80)
        X = X[self.selected_features]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        
        print(f"Final feature count: {len(self.selected_features)}")
        print("Features used:", self.selected_features)
        
        return X_scaled, y
        
    def train(self, X, y, n_splits=5):
        """Train multiple models with time series cross-validation"""
        print("Training models...")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            start_time = time.time()
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = r2_score(y_val, y_pred)
                scores.append(score)
                
                print(f"Fold score (R2): {score:.4f}")
            
            avg_score = np.mean(scores)
            training_time = time.time() - start_time
            
            results[model_name] = {
                'avg_r2': avg_score,
                'std_r2': np.std(scores),
                'training_time': training_time
            }
            
            print(f"{model_name} - Average R2: {avg_score:.4f}, Std: {np.std(scores):.4f}, Time: {training_time:.2f}s")
            
            # Update best model
            if avg_score > self.best_score:
                self.best_score = avg_score
                self.best_model_name = model_name
                self.best_model = model
        
        # Final fit with best model on all data
        print(f"\nBest model: {self.best_model_name} (R2: {self.best_score:.4f})")
        self.best_model.fit(X, y)
        
        # Store feature importance if available
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.selected_features,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 important features:")
            print(self.feature_importance.head(10))
        
        return results
        
    def predict(self, X):
        """Make predictions using the best model"""
        if self.best_model is None:
            raise ValueError("Models not trained yet")
            
        X = X[self.selected_features]
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)
        
    def evaluate(self, y_true, y_pred):
        """Evaluate model performance"""
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
        
        print("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
        return metrics

def main():
    """Main execution flow"""
    print("Starting CAD OAS prediction model training...")
    
    # Initialize predictor
    predictor = MLCADOASPredictor()
    
    # Load and preprocess data
    df, features = predictor.load_and_preprocess('pulling_data/backtest_data.csv')
    
    # Prepare features
    X, y = predictor.prepare_features(df, features)
    
    # Train models
    results = predictor.train(X, y)
    
    # Make predictions with best model
    predictions = predictor.predict(X)
    
    # Evaluate best model
    metrics = predictor.evaluate(y, predictions)
    
    # Print final comparison
    print("\nFinal Model Comparison:")
    comparison_df = pd.DataFrame(results).T
    print(comparison_df)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
