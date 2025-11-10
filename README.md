# Ensemble Learning for Bike Share Demand Prediction

A comprehensive implementation of ensemble learning techniques (Bagging, Boosting, and Stacking) for predicting bike rental demand using the UCI Bike Sharing Dataset.

## Author 
Name : Nikhil Enugu

Roll No: DA25M010

## 📋 Project Overview

This project demonstrates the application of three primary ensemble techniques to solve a complex time-series-based regression problem. The goal is to accurately forecast bike rentals by addressing model variance and bias through different ensemble strategies.

## 🎯 Objective

Predict the total count of rented bikes (`cnt`) using various ensemble methods and compare their effectiveness in minimizing prediction error (RMSE). The project explores how combining diverse models can yield superior performance compared to single models.

## 📊 Dataset

- **Source**: [UCI Machine Learning Repository - Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)
- **Size**: 17,379 hourly samples
- **Target Variable**: `cnt` (total count of bike rentals)
- **Features**: Weather conditions, temporal information (hour, day, month, season), and other environmental factors

**Citation**: Fanaee-T, Hadi, and Gamper, H. (2014). Bikeshare Data Set. UCI Machine Learning Repository.

## 🔍 Implementation Details

### Part A: Data Preprocessing and Baseline

1. **Feature Engineering**
   - Removed irrelevant columns (`instant`, `dteday`, `casual`, `registered`)
   - Applied One-Hot Encoding to categorical features (`season`, `weathersit`, `mnth`, `hr`, `weekday`)
   - Created 53 features after encoding

2. **Baseline Models**
   - Decision Tree Regressor (max_depth=6): RMSE = 118.46
   - Linear Regression: RMSE = 100.45 ✓ (Better baseline)

### Part B: Ensemble Techniques

1. **Bagging (Variance Reduction)**
   - Implementation: BaggingRegressor with 200 estimators
   - Base Estimator: Decision Tree Regressor
   - **Result**: RMSE = 112.27
   - **Analysis**: Successfully reduced variance compared to single Decision Tree (118.46 → 112.27)

2. **Boosting (Bias Reduction)**
   - Implementation: GradientBoostingRegressor with 250 estimators
   - **Result**: RMSE = 61.24
   - **Analysis**: Significant improvement through sequential error correction

### Part C: Stacking for Optimal Performance

**Architecture**:
- **Base Learners (Level-0)**:
  - K-Nearest Neighbors Regressor
  - Bagging Regressor (200 estimators)
  - Gradient Boosting Regressor (250 estimators)
  
- **Meta-Learner (Level-1)**: Ridge Regression (alpha=10)

- **Result**: RMSE = 58.29 ✓ (Best Performance)

## 📈 Results Comparison

| Model | RMSE | Rank |
|-------|------|------|
| **Stacking Regressor** | **58.29** | **🥇 1st** |
| Gradient Boosting Regressor | 61.24 | 🥈 2nd |
| Linear Regression | 100.45 | 3rd |
| Bagging Regressor | 112.27 | 4th |
| Decision Tree Regressor | 118.46 | 5th |

## 🎓 Key Findings

1. **Stacking Superiority**: The Stacking Regressor achieved the lowest RMSE (58.29) by effectively combining diverse model predictions through a meta-learner.

2. **Bias-Variance Trade-off**:
   - Bagging reduced variance in the high-variance Decision Tree model
   - Boosting addressed bias through sequential error correction
   - Stacking optimally balanced both bias and variance

3. **Model Diversity**: The success of stacking demonstrates the power of combining complementary learning behaviors from different algorithms.

## 📝 Conclusion

The project successfully demonstrates that:
- **Stacking outperforms** all individual and ensemble models by exploiting complementary learning behaviors
- **Ensemble diversity** is crucial for optimal generalization
- **Hierarchical combinations** (through meta-learning) can effectively balance bias and variance
- The 48.6% improvement over the baseline (118.46 → 58.29 RMSE) validates the power of ensemble methods
