import pandas as pd
import numpy as np
from pathlib import Path
from autots import AutoTS
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm

# Filter warnings but keep important ones
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def load_and_prepare_data():
    """
    Load and prepare data for ML modeling from backtest_data.csv
    Returns:
        pd.DataFrame: Processed dataframe ready for ML
    """
    print("Loading data...")
    data_path = Path(__file__).parent.parent / "pulling_data" / "backtest_data.csv"
    df = pd.read_csv(data_path)
    
    print("Processing data...")
    # Convert Date to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()
    
    # Handle missing and infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

def evaluate_model(model, validation_results, actual, predicted, prediction_bounds):
    """Enhanced evaluation function with comprehensive metrics and visualizations"""
    print("\nCalculating metrics...")
    metrics = {}
    
    try:
        # Get all available metrics from AutoTS
        metrics.update(validation_results.model_results.iloc[0].to_dict())
        
        # Calculate additional metrics
        metrics['SMAPE'] = np.mean(validation_results.model_results['smape'])
        metrics['RMSE'] = np.mean(validation_results.model_results['rmse'])
        metrics['MAE'] = np.mean(validation_results.model_results['mae'])
        
        print("\nGenerating visualizations...")
        # Create comprehensive visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Actual vs Predicted with prediction intervals
        ax1.plot(actual.index[-len(predicted):], actual[-len(predicted):], label='Actual', color='blue')
        ax1.plot(predicted.index, predicted, label='Predicted', color='red')
        if prediction_bounds is not None:
            ax1.fill_between(predicted.index, 
                            prediction_bounds.lower_forecast.values.flatten(),
                            prediction_bounds.upper_forecast.values.flatten(),
                            alpha=0.2, color='gray', label='90% Prediction Interval')
        ax1.set_title('Actual vs Predicted with Prediction Intervals')
        ax1.legend()
        
        # 2. Model Performance Metrics
        metrics_to_plot = ['SMAPE', 'RMSE', 'MAE']
        ax2.bar(metrics_to_plot, [metrics[m] for m in metrics_to_plot])
        ax2.set_title('Model Performance Metrics')
        
        # 3. Residuals Plot
        residuals = actual[-len(predicted):] - predicted
        sns.histplot(residuals, kde=True, ax=ax3)
        ax3.set_title('Residuals Distribution')
        
        # 4. Model Comparison
        model_metrics = validation_results.model_results[['smape', 'mae', 'rmse']]
        model_metrics.plot(kind='bar', ax=ax4)
        ax4.set_title('Model Comparison')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_path = Path(__file__).parent / "model_evaluation.png"
        plt.savefig(plot_path)
        plt.close()
        
        print(f"\nPlot saved to: {plot_path}")
        
    except Exception as e:
        print(f"Warning: Error during evaluation: {str(e)}")
        print("Continuing with available metrics...")
    
    return metrics

def build_prediction_model(df):
    """
    Build and train enhanced AutoTS model using multiple algorithms and comprehensive evaluation
    """
    print("\nInitializing model...")
    # Create the AutoTS model with enhanced parameters
    model = AutoTS(
        forecast_length=1,
        frequency='infer',
        prediction_interval=0.9,
        ensemble=['simple', 'horizontal-min', 'horizontal-max'],
        max_generations=4,
        num_validations=3,
        model_list=['LastValueNaive', 
                   'ETS',
                   'ARIMA', 
                   'GLM',
                   'DatepartRegression',
                   'WindowRegression'],
        transformer_list=['StandardScaler', 
                        'PowerTransformer',
                        'DifferencedTransformer',
                        'Detrend'],
        n_jobs=9
    )
    
    try:
        # Train the model with progress tracking
        print("\nTraining model...")
        print("This may take several minutes. Progress:")
        print("- Fitting models...")
        model = model.fit(df)
        
        print("- Generating predictions...")
        prediction = model.predict()
        forecast = prediction.forecast
        
        print("- Calculating validation scores...")
        validation_results = model.results()
        
        print("- Getting prediction bounds...")
        prediction_bounds = prediction
        
        return model, forecast, validation_results, prediction_bounds
        
    except Exception as e:
        print(f"\nError during model building: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        print("Starting model evaluation process...")
        print("=" * 50)
        
        df = load_and_prepare_data()
        
        print("\nDataset Information:")
        print(f"Shape: {df.shape}")
        print("\nFeatures:")
        print(df.columns.tolist())
        
        print("\nBuilding prediction model...")
        model, forecasts, validation_results, prediction_bounds = build_prediction_model(df)
        
        print("\nValidation Scores:")
        metrics = evaluate_model(model, validation_results, df, forecasts, prediction_bounds)
        
        # Print detailed results
        print("\nBest Model Metrics:")
        for metric, value in metrics.items():
            if metric in ['SMAPE', 'RMSE', 'MAE']:
                print(f"{metric}: {value:.4f}")
        
        print(f"\nBest Model: {model.best_model_name}")
        print(f"\nBest Model Parameters:")
        print(model.best_model_params)
        
        print("\nFinal Forecast:")
        print(forecasts)
        
        print("\nModel Summary:")
        print(f"Total Observations: {len(df)}")
        print(f"Training Period: {df.index[0]} to {df.index[-1]}")
        print(f"Forecast Period: {forecasts.index[0]}")
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        raise
