import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from hmmlearn import hmm

class MarketAnalyzer:
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
        self.target = 'cad_ig_er_index'
        
    def calculate_returns(self) -> pd.DataFrame:
        """Calculate returns and rolling statistics for the target"""
        returns = pd.DataFrame()
        returns['daily_ret'] = self.data[self.target].pct_change()
        returns['monthly_ret'] = self.data[self.target].pct_change(20)  # ~20 trading days
        returns['rolling_vol'] = returns['daily_ret'].rolling(20).std() * np.sqrt(252)
        returns['rolling_sharpe'] = (returns['daily_ret'].rolling(20).mean() * 252) / returns['rolling_vol']
        return returns
    
    def analyze_features(self) -> Dict:
        """Analyze feature relationships with target returns"""
        feature_stats = {}
        target_rets = self.data[self.target].pct_change()
        
        for col in self.data.columns:
            if col != self.target:
                # Calculate correlations at different lags
                correlations = []
                for lag in range(1, 11):
                    corr = target_rets.corr(self.data[col].shift(lag))
                    correlations.append((lag, corr))
                
                # Calculate information coefficient (rank correlation)
                ic = target_rets.corr(self.data[col].shift(1), method='spearman')
                
                feature_stats[col] = {
                    'correlations': correlations,
                    'ic': ic
                }
        
        return feature_stats
    
    def identify_regimes(self) -> pd.DataFrame:
        """Identify market regimes using various metrics"""
        regimes = pd.DataFrame(index=self.data.index)
        
        # VIX-based regime
        regimes['vix_regime'] = pd.qcut(self.data['vix'], q=3, labels=['low_vol', 'med_vol', 'high_vol'])
        
        # Spread regime
        regimes['spread_regime'] = pd.qcut(self.data['cad_oas'], q=3, labels=['tight', 'medium', 'wide'])
        
        # Economic regime (already provided)
        regimes['econ_regime'] = self.data['us_economic_regime']
        
        return regimes
    
    def calculate_regime_performance(self, regimes: pd.DataFrame) -> Dict:
        """Calculate performance metrics within each regime"""
        target_rets = self.data[self.target].pct_change()
        regime_stats = {}
        
        for regime_col in regimes.columns:
            stats = {}
            for regime in regimes[regime_col].unique():
                mask = regimes[regime_col] == regime
                regime_rets = target_rets[mask]
                
                stats[regime] = {
                    'mean_ret': regime_rets.mean() * 252,
                    'vol': regime_rets.std() * np.sqrt(252),
                    'sharpe': (regime_rets.mean() * 252) / (regime_rets.std() * np.sqrt(252)),
                    'win_rate': (regime_rets > 0).mean()
                }
            
            regime_stats[regime_col] = stats
            
        return regime_stats

    def generate_technical_features(self) -> pd.DataFrame:
        """Generate technical indicators for the target"""
        tech_features = pd.DataFrame(index=self.data.index)
        
        # Moving averages
        for window in [10, 20, 50]:
            tech_features[f'ma_{window}'] = self.data[self.target].rolling(window).mean()
            tech_features[f'ma_signal_{window}'] = (
                self.data[self.target] > tech_features[f'ma_{window}']
            ).astype(int)
            
        # RSI
        delta = self.data[self.target].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        tech_features['rsi'] = 100 - (100 / (1 + rs))
        
        # Momentum
        tech_features['momentum_20'] = self.data[self.target].pct_change(20)
        tech_features['momentum_50'] = self.data[self.target].pct_change(50)

        # MACD
        ema_12 = self.data[self.target].ewm(span=12, adjust=False).mean()
        ema_26 = self.data[self.target].ewm(span=26, adjust=False).mean()
        tech_features['macd'] = ema_12 - ema_26
        tech_features['macd_signal'] = tech_features['macd'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        tech_features['bollinger_mid'] = self.data[self.target].rolling(window=20).mean()
        tech_features['bollinger_std'] = self.data[self.target].rolling(window=20).std()
        tech_features['bollinger_upper'] = tech_features['bollinger_mid'] + (2 * tech_features['bollinger_std'])
        tech_features['bollinger_lower'] = tech_features['bollinger_mid'] - (2 * tech_features['bollinger_std'])

        # ATR (Average True Range)
        high = self.data[self.target].shift(1).fillna(self.data[self.target].iloc[0])
        low = self.data[self.target].shift(1).fillna(self.data[self.target].iloc[0])
        tech_features['true_range'] = np.maximum(self.data[self.target] - low, high - self.data[self.target])
        tech_features['atr'] = tech_features['true_range'].rolling(window=14).mean()

        # Interaction terms with VIX
        for feature in ['macd', 'rsi', 'momentum_20', 'momentum_50']:
            tech_features[f'{feature}_x_vix'] = tech_features[feature] * self.data['vix']

        # Lagged values
        for lag in range(1, 6):
            tech_features[f'target_lag_{lag}'] = self.data[self.target].shift(lag)
            tech_features[f'vix_lag_{lag}'] = self.data['vix'].shift(lag)

        # Time series decomposition (simple moving average for trend)
        tech_features['trend'] = self.data[self.target].rolling(window=20).mean()
        tech_features['seasonality'] = self.data[self.target] - tech_features['trend']

        return tech_features

    def test_stationarity(self, data: pd.Series, window: int = 20) -> pd.DataFrame:
        """Perform rolling Augmented Dickey-Fuller test."""
        adf_results = pd.DataFrame(index=data.index)
        adf_results['adf_statistic'] = data.rolling(window).apply(lambda x: adfuller(x)[0], raw=False)
        adf_results['p_value'] = data.rolling(window).apply(lambda x: adfuller(x)[1], raw=False)
        adf_results['critical_value_5'] = data.rolling(window).apply(lambda x: adfuller(x)[4]['5%'], raw=False)
        return adf_results

    def calculate_rolling_stats(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Calculate rolling correlations and other statistics."""
        rolling_stats = pd.DataFrame(index=data.index)
        rolling_stats['rolling_mean'] = data[self.target].rolling(window).mean()
        rolling_stats['rolling_std'] = data[self.target].rolling(window).std()

        for col in data.columns:
            if col != self.target:
                rolling_stats[f'rolling_corr_{col}'] = data[self.target].rolling(window).corr(data[col])

        return rolling_stats

    def perform_cluster_analysis(self, data: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        """Perform KMeans cluster analysis to identify market states."""
        # Select features for clustering
        cluster_features = ['cad_ig_er_index', 'vix', 'cad_oas', 'us_economic_regime']
        cluster_data = data[cluster_features].copy()

        # Handle categorical variable 'us_economic_regime' by one-hot encoding
        cluster_data = pd.get_dummies(cluster_data, columns=['us_economic_regime'], drop_first=True)

        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init = 'auto')
        clusters = kmeans.fit_predict(scaled_data)

        # Add cluster labels to the data
        cluster_results = pd.DataFrame(index=data.index)
        cluster_results['cluster'] = clusters

        return cluster_results

    def perform_cluster_analysis_minmax(self, data: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        """Perform KMeans cluster analysis with MinMaxScaler to identify market states."""
        # Select features for clustering
        cluster_features = ['cad_ig_er_index', 'vix', 'cad_oas', 'us_economic_regime']
        cluster_data = data[cluster_features].copy()

        # Handle categorical variable 'us_economic_regime' by one-hot encoding
        cluster_data = pd.get_dummies(cluster_data, columns=['us_economic_regime'], drop_first=True)

        # Scale the data using MinMaxScaler
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(cluster_data)

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init = 'auto')
        clusters = kmeans.fit_predict(scaled_data)

        # Add cluster labels to the data
        cluster_results = pd.DataFrame(index=data.index)
        cluster_results['cluster_minmax'] = clusters

        return cluster_results

    def identify_regimes_hmm(self, data: pd.DataFrame, n_components: int = 3) -> pd.DataFrame:
        """Identify market regimes using Hidden Markov Models."""
        # Select features for HMM
        hmm_features = ['cad_ig_er_index', 'vix', 'cad_oas']
        hmm_data = data[hmm_features].copy()

        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(hmm_data)

        # Fit HMM
        model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", random_state=42)
        model.fit(scaled_data)
        regimes = model.predict(scaled_data)

        # Add regime labels to the data
        regime_results = pd.DataFrame(index=data.index)
        regime_results['hmm_regime'] = regimes

        return regime_results

    def generate_summary_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics for all features, including handling missing values and outliers."""
        summary_stats = pd.DataFrame(index=data.columns)
        summary_stats['dtype'] = data.dtypes
        summary_stats['count'] = data.count()
        summary_stats['missing'] = data.isnull().sum()
        summary_stats['missing_pct'] = data.isnull().sum() / len(data) * 100
        summary_stats['mean'] = data.mean()
        summary_stats['std'] = data.std()
        summary_stats['min'] = data.min()
        summary_stats['max'] = data.max()

        # Calculate outlier counts (using IQR method)
        outlier_counts = {}
        for col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            outlier_counts[col] = outlier_count

        summary_stats['outliers'] = pd.Series(outlier_counts)

        return summary_stats

def main():
    analyzer = MarketAnalyzer('pulling_data/backtest_data.csv')
    
    # Calculate basic returns and statistics
    returns = analyzer.calculate_returns()
    
    # Analyze feature relationships
    feature_stats = analyzer.analyze_features()
    
    # Generate technical features
    tech_features = analyzer.generate_technical_features()

    # Test for stationarity
    adf_results = analyzer.test_stationarity(analyzer.data[analyzer.target])

    # Calculate rolling statistics
    rolling_stats = analyzer.calculate_rolling_stats(analyzer.data)

    # Identify regimes using HMM
    hmm_regimes = analyzer.identify_regimes_hmm(analyzer.data)

    # Perform cluster analysis
    cluster_results = analyzer.perform_cluster_analysis(analyzer.data)

    # Perform cluster analysis with MinMaxScaler
    cluster_results_minmax = analyzer.perform_cluster_analysis_minmax(analyzer.data)

    # Generate summary statistics
    summary_stats = analyzer.generate_summary_statistics(analyzer.data)

    # Print key findings
    print("\n=== Feature Information Coefficients ===")
    for feature, stats in feature_stats.items():
        print(f"{feature}: IC = {stats['ic']:.3f}")
    
    print("\n=== HMM Regime Identification ===")
    print(hmm_regimes)

    print("\n=== ADF Test Results (Last 20 days) ===")
    print(adf_results.tail(20))

    print("\n=== Rolling Statistics (Last 20 days, subset of columns) ===")
    print(rolling_stats[['rolling_mean', 'rolling_std', 'rolling_corr_vix']].tail(20))

    print("\n=== Cluster Analysis Results ===")
    print(cluster_results)

    print("\n=== Cluster Analysis Results with MinMaxScaler ===")
    print(cluster_results_minmax)

    print("\n=== Summary Statistics ===")
    print(summary_stats)

if __name__ == "__main__":
    main()
