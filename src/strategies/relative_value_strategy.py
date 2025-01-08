"""Enhanced Relative Value Strategy Implementation."""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from itertools import product
import vectorbt as vbt
from scipy import stats

from .strategy_base import StrategyBase


class RelativeValueStrategy(StrategyBase):
    """Enhanced strategy combining relative value, trend, and regime signals."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize strategy with configuration."""
        super().__init__(config)
        self.name = "Enhanced RV"
        
        # Strategy parameters with defaults
        self.params = {
            # Relative Value Parameters
            'zscore_window': 24,
            'zscore_entry': 0.5,
            'zscore_exit': -0.25,
            
            # Trend Parameters
            'fast_ma': 5,
            'slow_ma': 20,
            'trend_threshold': 0.0,
            
            # Volatility Parameters
            'vol_window': 12,
            'vol_threshold': 2.0,
            
            # Momentum Parameters
            'momentum_window': 3,
            'min_momentum': -0.02,
            
            # Regime Parameters
            'regime_window': 60,
            'regime_threshold': 0.0,
            
            # Mean Reversion Parameters
            'mean_rev_window': 10,
            'mean_rev_threshold': 1.0,
            
            # Position Sizing
            'base_position_size': 1.0,
            'momentum_scale': 1.0,
            'trend_scale': 1.0,
            'vol_scale': 1.0
        }
        
        # Update parameters from config if provided
        if 'relative_value' in config:
            self.params.update(config['relative_value'])
    
    def detect_frequency_and_set_ranges(self, df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """Detect data frequency and set appropriate parameter ranges."""
        # Calculate average days between data points
        freq_days = (df.index[-1] - df.index[0]).days / len(df)
        
        # Determine if data is daily, weekly, monthly, etc.
        if freq_days <= 5:  # Daily
            base_window = 21  # ~1 month
        elif freq_days <= 10:  # Weekly
            base_window = 4   # ~1 month
        else:  # Monthly or longer
            base_window = 3   # 3 months
            
        # Set parameter ranges based on frequency - now even more aggressive
        param_ranges = {
            # Windows scaled by base_window - shorter windows for faster signals
            'zscore_window': (base_window, base_window * 4),    # 3-12 months for monthly
            'vol_window': (1, base_window),                     # 1-3 months for monthly
            'momentum_window': (1, base_window),                # 1-3 months for monthly
            'mean_rev_window': (1, base_window),                # 1-3 months for monthly
            'fast_ma': (1, base_window),                        # 1-3 months for monthly
            'slow_ma': (base_window, base_window * 4),          # 3-12 months for monthly
            
            # More extreme thresholds for more frequent signals
            'zscore_entry': (0.0, 4.0),                        # Allow even more extreme entries
            'zscore_exit': (-4.0, 0.0),                        # Allow even more extreme exits
            'trend_threshold': (-0.1, 0.1),                    # Even wider trend range
            'vol_threshold': (1.0, 10.0),                      # Much more tolerant of volatility
            'min_momentum': (-0.1, 0.1),                       # Even wider momentum range
            'mean_rev_threshold': (0.5, 4.0),                  # More extreme mean reversion
            
            # Super aggressive position sizing
            'base_position_size': (1.0, 2.0),                  # Minimum 100% position size
            'momentum_scale': (0.5, 3.0),                      # Up to 300% scaling on momentum
            'trend_scale': (0.5, 3.0),                         # Up to 300% scaling on trend
            'vol_scale': (0.5, 3.0)                           # Up to 300% scaling on volatility
        }
        
        return param_ranges, freq_days

    def calculate_regime_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate regime score based on multiple factors."""
        # Market stress from VIX
        vix_mean = df['vix'].rolling(24).mean()
        vix_std = df['vix'].rolling(24).std()
        vix_zscore = (df['vix'] - vix_mean) / vix_std
        vix_signal = (vix_zscore < 1.0).astype(float)
        
        # Yield curve slope
        curve_signal = (df['us_3m_10y'] > 0).astype(float)
        
        # Combine signals
        regime_score = (vix_signal + curve_signal) / 2
        return regime_score
    
    def calculate_signals_with_metrics(self, df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.Series, Dict[str, Any]]:
        """Calculate trading signals using multiple factors."""
        # Convert window parameters to integers
        window_params = ['zscore_window', 'vol_window', 'momentum_window', 'mean_rev_window', 'fast_ma', 'slow_ma']
        for param in window_params:
            params[param] = int(round(params[param]))
        
        # Initialize signals series with float dtype
        signals = pd.Series(1.0, index=df.index)  # Start with base position
        
        try:
            # 1. Relative Value Signal with exponential weighting
            spread = df['cad_oas'] - df['us_ig_oas']
            spread_mean = spread.ewm(span=params['zscore_window'], min_periods=1).mean()
            spread_std = spread.ewm(span=params['zscore_window'], min_periods=1).std()
            zscore = (spread - spread_mean) / spread_std.replace(0, 1)
            
            # 2. Enhanced Trend Signal with exponential weighting
            fast_ma = df['cad_ig_er_index'].ewm(span=params['fast_ma'], min_periods=1).mean()
            slow_ma = df['cad_ig_er_index'].ewm(span=params['slow_ma'], min_periods=1).mean()
            trend_strength = (fast_ma - slow_ma) / slow_ma.replace(0, 1)
            trend_signal = trend_strength > params['trend_threshold']
            
            # 3. Volatility Signal with exponential weighting
            returns = df['cad_ig_er_index'].pct_change().fillna(0)
            volatility = returns.ewm(span=params['vol_window'], min_periods=1).std() * np.sqrt(12)
            vol_mean = volatility.ewm(span=24, min_periods=1).mean()
            vol_std = volatility.ewm(span=24, min_periods=1).std()
            volatility_zscore = (volatility - vol_mean) / vol_std.replace(0, 1)
            vol_filter = volatility_zscore < params['vol_threshold']
            
            # 4. Enhanced Momentum Signal with exponential weighting
            momentum = returns.ewm(span=params['momentum_window'], min_periods=1).mean() * params['momentum_window']
            momentum_signal = momentum > params['min_momentum']
            momentum_std = momentum.ewm(span=params['momentum_window'], min_periods=1).std()
            momentum_strength = momentum / momentum_std.replace(0, 1)
            
            # 5. Regime Signal - Super aggressive
            regime_score = self.calculate_regime_score(df)
            regime_filter = regime_score > 0.2  # Even more lenient regime filter
            
            # 6. Mean Reversion Signal with exponential weighting
            returns_mean = returns.ewm(span=params['mean_rev_window'], min_periods=1).mean()
            returns_std = returns.ewm(span=params['mean_rev_window'], min_periods=1).std()
            returns_zscore = (returns - returns_mean) / returns_std.replace(0, 1)
            mean_rev_signal = returns_zscore < -params['mean_rev_threshold']
            
            # Combine Signals - More aggressive combination
            rv_entry = zscore > params['zscore_entry']
            rv_exit = zscore < params['zscore_exit']
            
            # Super aggressive signal combination
            entry_signals = (
                rv_entry |  # Relative value entry
                (mean_rev_signal & (trend_signal | momentum_signal)) |  # Mean reversion with either trend or momentum
                (momentum_signal & trend_signal) |  # Momentum with trend confirmation
                (abs(zscore) > 2.0) |  # Take extreme opportunities
                (abs(trend_strength) > 0.05) |  # Strong trends
                (abs(momentum_strength) > 2.0)  # Strong momentum
            )
            
            # Apply filters - super lenient
            entry_signals = entry_signals & (vol_filter | regime_filter | momentum_signal)  # Need only one filter to pass
            
            # Dynamic position sizing with aggressive scaling
            position_size = params['base_position_size']
            
            # Scale position size based on signal strength with more aggressive scaling
            momentum_scale = np.clip(abs(momentum_strength), 0.5, 3.0) * params['momentum_scale']
            trend_scale = np.clip(abs(trend_strength), 0.5, 3.0) * params['trend_scale']
            vol_scale = np.clip(2.0 / (abs(volatility_zscore) + 1), 0.5, 3.0) * params['vol_scale']
            
            # Combine scales with aggressive weighting
            signal_strength = (
                momentum_scale * 0.4 +  # More weight on momentum
                trend_scale * 0.4 +     # More weight on trend
                vol_scale * 0.2         # Less weight on volatility
            )
            
            # Apply position sizing with aggressive scaling
            signals[entry_signals] = position_size * np.clip(signal_strength[entry_signals], 0.5, 3.0)
            signals[rv_exit] = 0.0
            
            # Allow maximum leverage
            signals = signals.clip(0, 3.0)  # Allow up to 300% position size
            
            # Calculate diagnostic metrics
            metrics = {
                'market_exposure': signals.mean(),
                'avg_holding_period': signals.groupby((signals != signals.shift()).cumsum()).size().mean(),
                'zscore_opportunities': (zscore > params['zscore_entry']).mean(),
                'trend_signals': trend_signal.mean(),
                'mean_rev_signals': mean_rev_signal.mean(),
                'vol_filter_pass': vol_filter.mean(),
                'regime_filter_pass': regime_filter.mean(),
                'combined_opportunities': entry_signals.mean(),
                'avg_position_size': signals[signals > 0].mean()
            }
            
        except Exception as e:
            print(f"Error in signal generation: {str(e)}")
            signals = pd.Series(0.0, index=df.index)
            metrics = {
                'market_exposure': 0.0,
                'avg_holding_period': 0.0,
                'zscore_opportunities': 0.0,
                'trend_signals': 0.0,
                'mean_rev_signals': 0.0,
                'vol_filter_pass': 0.0,
                'regime_filter_pass': 0.0,
                'combined_opportunities': 0.0,
                'avg_position_size': 0.0
            }
        
        return signals, metrics
    
    def optimize_parameters(self, price_series: pd.Series) -> Tuple[Dict[str, Any], pd.Series]:
        """Find optimal parameters through genetic algorithm optimization."""
        # Load full dataset
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        data_path = os.path.join(project_root, 'pulling_data', 'backtest_data.csv')
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Get parameter ranges based on data frequency
        param_ranges, freq_days = self.detect_frequency_and_set_ranges(df)
        print(f"\nDetected data frequency: {freq_days:.1f} days")
        print("Parameter ranges adjusted for frequency:")
        for param, (min_val, max_val) in param_ranges.items():
            print(f"{param}: {min_val:.3f} to {max_val:.3f}")
        
        def create_individual():
            """Create a random individual (parameter set)."""
            return {k: np.random.uniform(v[0], v[1]) for k, v in param_ranges.items()}
        
        def fitness(params):
            """Calculate fitness (total return) for a parameter set."""
            try:
                signals, _ = self.calculate_signals_with_metrics(df, params)
                portfolio = vbt.Portfolio.from_signals(
                    price_series,
                    signals,
                    init_cash=100,
                    freq='1D'
                )
                total_return = portfolio.total_return()
                if isinstance(total_return, (pd.Series, pd.DataFrame)):
                    total_return = total_return.iloc[0]
                return float(total_return)
            except Exception as e:
                print(f"Error in fitness calculation: {str(e)}")
                return -np.inf
        
        # Genetic Algorithm Parameters
        population_size = 50
        generations = 20
        elite_size = 5
        mutation_rate = 0.1
        
        # Initialize population
        population = [create_individual() for _ in range(population_size)]
        
        print("\nOptimizing Enhanced RV Strategy using Genetic Algorithm...")
        
        for generation in range(generations):
            # Evaluate fitness for all individuals
            fitness_scores = [(params, fitness(params)) for params in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Print progress
            best_score = fitness_scores[0][1]
            print(f"Generation {generation + 1}/{generations}, Best Return: {best_score * 100:.2f}%")
            
            # Select elite individuals
            new_population = [params for params, _ in fitness_scores[:elite_size]]
            
            # Create rest of new population
            while len(new_population) < population_size:
                # Tournament selection
                tournament = np.random.choice(population, size=3)
                parent1 = max(tournament, key=fitness)
                tournament = np.random.choice(population, size=3)
                parent2 = max(tournament, key=fitness)
                
                # Crossover
                child = {}
                for param in param_ranges.keys():
                    if np.random.random() < 0.5:
                        child[param] = parent1[param]
                    else:
                        child[param] = parent2[param]
                
                # Mutation
                for param in param_ranges.keys():
                    if np.random.random() < mutation_rate:
                        min_val, max_val = param_ranges[param]
                        child[param] = np.random.uniform(min_val, max_val)
                
                new_population.append(child)
            
            population = new_population
        
        # Get best parameters
        best_params = max(population, key=fitness)
        best_signals, best_metrics = self.calculate_signals_with_metrics(df, best_params)
        
        print("\nOptimization complete!")
        print(f"\nBest parameters found:")
        for param, value in best_params.items():
            print(f"{param}: {value:.3f}")
        
        total_return = fitness(best_params)
        print(f"Best total return: {total_return * 100:.2f}%")
        
        print("\nStrategy Metrics for Best Parameters:")
        print(f"Market Exposure: {best_metrics['market_exposure']*100:.1f}%")
        print(f"Average Holding Period: {best_metrics.get('avg_holding_period', 0):.1f} days")
        
        print("\nSignal Breakdown:")
        print(f"Z-score opportunities: {best_metrics['zscore_opportunities']*100:.1f}%")
        print(f"Trend signals active: {best_metrics['trend_signals']*100:.1f}%")
        print(f"Mean reversion active: {best_metrics['mean_rev_signals']*100:.1f}%")
        print(f"After volatility filter: {best_metrics['vol_filter_pass']*100:.1f}%")
        print(f"After regime filter: {best_metrics['regime_filter_pass']*100:.1f}%")
        print(f"Combined opportunities: {best_metrics['combined_opportunities']*100:.1f}%")
        
        # Update strategy parameters with optimal values
        self.params = best_params
        
        return best_params, best_signals
    
    def generate_signals(self, price_series: pd.Series) -> pd.Series:
        """Generate trading signals based on multiple factors."""
        # Load full dataset since we need additional data
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        data_path = os.path.join(project_root, 'pulling_data', 'backtest_data.csv')
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Optimize parameters if not already optimized
        if not hasattr(self, 'optimized'):
            _, signals = self.optimize_parameters(price_series)
            self.optimized = True
            return signals
        
        # Calculate signals using current parameters
        signals, _ = self.calculate_signals_with_metrics(df, self.params)
        return signals
