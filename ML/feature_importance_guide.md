# Feature Importance Analysis Guide

This document provides a comprehensive guide to interpreting the feature importance analysis results from our credit spread prediction model.

## 1. Individual Model Feature Importances

Each model (LightGBM, Random Forest, ElasticNet) calculates feature importance differently:

- **LightGBM**: Based on the gain in the split criterion when using the feature
- **Random Forest**: Based on the decrease in impurity (MSE for regression)
- **ElasticNet**: Based on the absolute values of the coefficients

The normalized importance scores allow for comparison across models.

## 2. Feature Selection

Features are selected based on their aggregate importance across all models. The threshold-based selection:
- Keeps features with mean importance > threshold (default: 0.01)
- Helps reduce dimensionality while maintaining predictive power
- Should be adjusted based on the specific use case and data characteristics

## 3. Feature Stability Analysis

Stability analysis helps understand how consistent feature importances are across different time periods:

- **Mean Importance**: Average importance across time periods
- **Standard Deviation**: Variation in importance
- **Coefficient of Variation (CV)**: Relative variability (std/mean)
- **Min/Max**: Range of importance values

Lower CV values indicate more stable feature importance.

## 4. Feature Interactions

Interaction analysis reveals how features work together:

- **Interaction Score**: Measures the additional predictive power gained by combining features
- **Positive Score**: Features complement each other
- **Negative Score**: Features may be redundant

### Interpreting Results

1. **Strong, Stable Features**:
   - High mean importance
   - Low CV
   - Good candidates for core model features

2. **Strong but Unstable Features**:
   - High mean importance
   - High CV
   - May need special handling or regime-specific models

3. **Strong Interactions**:
   - High interaction scores
   - Consider creating interaction features
   - May indicate regime-dependent relationships

## Best Practices

1. **Feature Selection**:
   - Start with stable, important features
   - Add interaction features for strong interactions
   - Consider regime-specific features for unstable relationships

2. **Model Monitoring**:
   - Track feature importance stability over time
   - Watch for shifts in feature relationships
   - Update feature set based on stability metrics

3. **Model Improvement**:
   - Focus feature engineering on stable, important features
   - Create interaction features for complementary features
   - Consider separate models for different regimes if features are unstable

## Warning Signs

1. **Unstable Features**:
   - High CV across time periods
   - May indicate overfitting or regime changes

2. **Weak Interactions**:
   - Low interaction scores
   - May indicate redundant features

3. **Inconsistent Importance**:
   - Large differences between models
   - May indicate model-specific biases or data quality issues
