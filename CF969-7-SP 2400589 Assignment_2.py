# Import essential libraries for data handling, modeling, and visualization
import yfinance as yf                    # For downloading financial data
import pandas as pd                      # For handling tabular data
import numpy as np                       # For numerical operations
import matplotlib.pyplot as plt          # For plotting
import seaborn as sns                    # For enhanced visualizations

from scipy.stats import zscore           # For outlier detection
from sklearn.linear_model import LinearRegression             # Linear regression model
from sklearn.svm import SVR                                # Support Vector Regression
from sklearn.ensemble import RandomForestRegressor          # Random Forest Regressor
from sklearn.preprocessing import StandardScaler            # Standardize features
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score  # Model selection tools
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Evaluation metrics
from sklearn.feature_selection import RFE                   # Recursive Feature Elimination for feature selection

import tensorflow as tf                                    # TensorFlow for building Neural Networks
from tensorflow.keras.models import Sequential             # Sequential NN model
from tensorflow.keras.layers import Dense                  # Dense (fully connected) layers
from tensorflow.keras.callbacks import EarlyStopping       # Stop training early to avoid overfitting

import warnings
warnings.filterwarnings("ignore") # Suppress warnings for clean output

# Define the list of stock tickers to analyze
tickers = ['AAPL', 'GOOGL', 'AMZN', 'NVDA', 'QCOM']

# Define the market index (S&P 500) ticker
index_ticker = '^GSPC'

# Set start and end dates for 5 years of data
start_date = "2020-03-02"
end_date = "2025-03-02"

# Initialize a dictionary to store stock data
data = {}

# Loop through each ticker and download historical data using yfinance
for ticker in tickers:
    df_ticker = yf.download(ticker, start=start_date, end=end_date)

    if not df_ticker.empty:
        # Store selected columns after dropping missing values
        data[ticker] = df_ticker[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    else:
        print(f"Failed to download data for {ticker}")  # Handle download issues

# Download S&P 500 index data to serve as a benchmark
index_df = yf.download(index_ticker, start=start_date, end=end_date)

if not index_df.empty:
    # Only keep the 'Close' price of the index
    data[index_ticker] = index_df[['Close']].dropna()
else:
    raise ValueError("Failed to retrieve S&P 500 data.")  # Raise error if index data fails

# Combine all individual stock and index data into a single MultiIndex DataFrame
df = pd.concat(data, axis=1)

# Drop any rows with missing values across any column
df.dropna(inplace=True)

# Print the final shape of the dataset
print("Final dataset shape:", df.shape)


# Feature Engineering (Optimized)
# Function to compute engineered features for a stock using both technical and market indicators
def compute_features(df, stock, index):
    close = df[stock]['Close']              # Stock's daily close price
    volume = df[stock]['Volume']            # Trading volume
    index_close = df[index]['Close']        # S&P 500 closing prices (for market-based indicators)

    returns = close.pct_change()            # Daily returns as percentage change
    features = pd.DataFrame(index=returns.index)  # Initialize a new DataFrame to store features

    # Define target variable: next day's return (i.e., return_t+1)
    features['return'] = returns.shift(-1)

    # --- Technical Indicators (Features) ---
    features['daily_return'] = returns                             # Current day's return
    features['high_low_range'] = (df[stock]['High'] - df[stock]['Low']) / close  # Volatility proxy
    features['volatility'] = returns.rolling(20).std()             # Rolling standard deviation (20-day window)
    features['volume'] = volume                                     # Daily volume

    # --- Moving Averages ---
    features['ma10'] = close.rolling(window=10).mean().pct_change()     # Short-term trend
    features['ma50'] = close.rolling(window=50).mean().pct_change()     # Medium-term trend
    features['ma200'] = close.rolling(window=200).mean().pct_change()   # Long-term trend

    # --- RSI Calculation (Relative Strength Index) ---
    delta = close.diff()                            # Daily price difference
    up = delta.clip(lower=0)                        # Keep only gains
    down = -delta.clip(upper=0)                     # Keep only losses
    avg_gain = up.rolling(14).mean()                # 14-day average gain
    avg_loss = down.rolling(14).mean()              # 14-day average loss
    rs = avg_gain / avg_loss                        # Relative strength
    features['RSI'] = 100 - (100 / (1 + rs))        # RSI formula

    # --- Market-based Indicators (from S&P 500) ---
    features['index_return'] = index_close.pct_change()                       # S&P 500 daily return
    features['sp500_volatility'] = features['index_return'].rolling(20).std() # Rolling market volatility

    features.dropna(inplace=True)  # Remove rows with any missing values (due to rolling calculations)

    return features


# Apply the compute_features function to each stock
stock_data = {}
for ticker in tickers:
    stock_data[ticker] = compute_features(df, ticker, index_ticker)

# Create dictionaries to hold train/test splits for each stock
X_train, X_test, y_train, y_test = {}, {}, {}, {}

for ticker in tickers:
    data = stock_data[ticker].copy()  # Work with a copy to avoid modifying original

    # --- Outlier Removal using Z-Score ---
    z_scores = np.abs(zscore(data))
    data = data[(z_scores < 3).all(axis=1)]  # Keep rows where all z-scores < 3

    # Split features and target
    X = data.drop(columns=['return'])  # Features
    y = data['return']                 # Target variable (next day's return)

    # --- Split 80% training, 20% testing ---
    split_index = int(len(X) * 0.8)
    X_train[ticker], X_test[ticker] = X.iloc[:split_index], X.iloc[split_index:]
    y_train[ticker], y_test[ticker] = y.iloc[:split_index], y.iloc[split_index:]

    # --- Feature Standardization (Z-score normalization) ---
    scaler = StandardScaler()
    X_train[ticker] = scaler.fit_transform(X_train[ticker])  # Fit and transform training data
    X_test[ticker] = scaler.transform(X_test[ticker])        # Only transform test data


cv_results = []  # To store all metrics for summary

# Define 5-fold cross-validator with shuffle
cv = KFold(n_splits=5, shuffle=True, random_state=42)

for ticker in tickers:
    print(f"\nCross-Validation for {ticker}")
    X = X_train[ticker]  # Training features
    y = y_train[ticker]  # Training target

    # Define 3 base models
    models = {
        'Linear Regression': LinearRegression(),
        'SVM': SVR(kernel='linear', C=0.1, gamma=0.001),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    # Loop over each model
    for name, model in models.items():
        # Cross-validated MSE (lower = better)
        mse_scores = -1 * cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')

        # Cross-validated MAE (lower = better)
        mae_scores = -1 * cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')

        # Cross-validated R² Score (closer to 1 = better)
        r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

        # Print formatted output
        print(f"{name}: MSE={mse_scores.mean():.6f} ± {mse_scores.std():.6f}, "
              f"MAE={mae_scores.mean():.6f} ± {mae_scores.std():.6f}, "
              f"R²={r2_scores.mean():.4f} ± {r2_scores.std():.4f}")

        # Save results in a dictionary for DataFrame later
        cv_results.append({
            'Stock': ticker,
            'Model': name,
            'MSE (CV Mean)': mse_scores.mean(),
            'MAE (CV Mean)': mae_scores.mean(),
            'R2 (CV Mean)': r2_scores.mean(),
            'MSE Std': mse_scores.std(),
            'MAE Std': mae_scores.std(),
            'R2 Std': r2_scores.std()
        })

# Convert results to DataFrame
cv_results_df = pd.DataFrame(cv_results)

# Show sorted summary
print("\nCross-Validation Summary")
print(cv_results_df.sort_values(by=['Stock', 'MSE (CV Mean)']))

# Set Seaborn theme
sns.set(style='whitegrid')

# R² Scores Across Models and Stocks
plt.figure(figsize=(12, 6))
sns.barplot(data=cv_results_df, x='Stock', y='R2 (CV Mean)', hue='Model')
plt.title("Cross-Validation R² Scores by Model and Stock")
plt.ylabel("Average R² Score")
plt.grid(True)
plt.legend(title='Model')
plt.tight_layout()

# MAE Scores Across Models and Stocks
plt.figure(figsize=(12, 6))
sns.barplot(data=cv_results_df, x='Stock', y='MAE (CV Mean)', hue='Model')
plt.title("Cross-Validation MAE by Model and Stock")
plt.ylabel("Average Mean Absolute Error")
plt.grid(True)
plt.legend(title='Model')
plt.tight_layout()

# MSE Scores Across Models and Stocks
plt.figure(figsize=(12, 6))
sns.barplot(data=cv_results_df, x='Stock', y='MSE (CV Mean)', hue='Model')
plt.title("Cross-Validation MSE by Model and Stock")
plt.ylabel("Average Mean Squared Error")
plt.grid(True)
plt.legend(title='Model')
plt.tight_layout()

# Define a function to plot various features for each stock
def plot_stock_features_all(stock_data, tickers):
    # List of technical and market-based features to plot
    features_to_plot = [
        'daily_return',          # Daily percentage change in price
        'high_low_range',        # Intraday price volatility
        'volatility',            # 20-day rolling standard deviation of returns
        'volume',                # Trading volume
        'ma10', 'ma50', 'ma200', # 10, 50, 200-day moving average returns
        'RSI',                   # Relative Strength Index (momentum indicator)
        'index_return',          # S&P 500 daily return
        'sp500_volatility'       # 20-day volatility of the S&P 500
    ]

    # Loop through each stock passed to the function
    for stock in tickers:
        print(f"\nPlotting features for: {stock}")  # Print status for tracking

        # Create a new figure for each stock
        plt.figure(figsize=(14, 10))  # Set size large enough for 10 subplots

        # Enumerate over each feature to create subplots
        for i, feature in enumerate(features_to_plot, 1):
            plt.subplot(5, 2, i)  # Arrange plots in 5 rows x 2 columns
            plt.plot(stock_data[stock][feature])  # Plot the feature's time series
            plt.title(f"{stock} - {feature}")     # Title for clarity

        # Automatically adjust subplot spacing to prevent overlap
        plt.tight_layout()

# Call the function to generate plots for AAPL (can be expanded to all stocks)
plot_stock_features_all(stock_data, ['AAPL'])

# Feature Selection using RFE
for ticker in tickers:
    print(f"\n--- Feature Selection for {ticker} ---")

    estimator = LinearRegression()  # Base model used for RFE
    selector = RFE(estimator, n_features_to_select=5, step=1)  # Select top 5 features

    # Use original (unscaled) feature data for selection
    original_X = stock_data[ticker].drop(columns=['return'])  # Features
    original_y = stock_data[ticker]['return']  # Target return

    # Fit selector
    selector = selector.fit(StandardScaler().fit_transform(original_X), original_y)

    # Extract names of selected features
    original_features = original_X.columns
    selected_features = original_features[selector.support_]
    print(f"Selected features for {ticker}: {selected_features}")

    # Apply selection to the already standardized train/test sets
    X_train[ticker] = selector.transform(X_train[ticker])
    X_test[ticker] = selector.transform(X_test[ticker])


# SVM Hyperparameter Tuning

# --- SVM Parameter Grid for GridSearchCV ---
param_grid_svm = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

# Model Training and Evaluation
results = []  # To store MSE, MAE, R² for all models and stocks

# Dictionaries to store trained models
lr_models = {}   # Linear Regression
rf_models = {}   # Random Forest
nn_models = {}   # Neural Network
svm_models = {}  # SVM

# Lists to collect directional accuracy and Sharpe Ratio results
directional_results = []
sharpe_results = []

# Loop through each stock ticker
for ticker in tickers:
    print(f"\n--- {ticker} ---")

    # --- Utility Functions ---

    # Function to calculate directional accuracy (matching signs)
    def directional_accuracy(y_true, y_pred):
        return np.mean(np.sign(y_true) == np.sign(y_pred))

    # Function to compute Sharpe Ratio for predicted returns
    def sharpe_ratio(predicted_returns):
        mean_return = np.mean(predicted_returns)
        std_return = np.std(predicted_returns)
        return (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0

    # --- Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train[ticker], y_train[ticker])  # Fit model on training data
    lr_pred = lr.predict(X_test[ticker])      # Predict on test data

    # Calculate metrics
    dir_acc_lr = directional_accuracy(y_test[ticker].values, lr_pred)
    sr = sharpe_ratio(lr_pred)
    lr_mse = mean_squared_error(y_test[ticker], lr_pred)
    lr_mae = mean_absolute_error(y_test[ticker], lr_pred)
    lr_r2 = r2_score(y_test[ticker], lr_pred)

    # Store results
    directional_results.append([ticker, 'Linear Regression', dir_acc_lr])
    sharpe_results.append([ticker, 'Linear Regression', sr])
    results.append([ticker, 'Linear Regression', lr_mse, lr_mae, lr_r2])
    lr_models[ticker] = lr  # Save the trained model

    # --- SVM with Hyperparameter Tuning using GridSearchCV ---
    svm = SVR()
    grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='neg_mean_squared_error', verbose=1)
    grid_search_svm.fit(X_train[ticker], y_train[ticker])  # Perform grid search on training set
    best_svm = grid_search_svm.best_estimator_  # Get best model from search
    svm_pred = best_svm.predict(X_test[ticker])  # Predict on test set

    # Metrics for SVM
    dir_acc_svm = directional_accuracy(y_test[ticker].values, svm_pred)
    sr = sharpe_ratio(svm_pred)
    svm_mse = mean_squared_error(y_test[ticker], svm_pred)
    svm_mae = mean_absolute_error(y_test[ticker], svm_pred)
    svm_r2 = r2_score(y_test[ticker], svm_pred)

    # Save results
    directional_results.append([ticker, 'SVM', dir_acc_svm])
    sharpe_results.append([ticker, 'SVM', sr])
    results.append([ticker, 'SVM', svm_mse, svm_mae, svm_r2])
    print(f"Best SVM parameters for {ticker}: {grid_search_svm.best_params_}")
    svm_models[ticker] = best_svm  # Save trained model

    # --- Random Forest Regressor ---
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train[ticker], y_train[ticker])  # Train RF
    rf_pred = rf.predict(X_test[ticker])      # Predict on test data

    # Metrics for Random Forest
    dir_acc_rf = directional_accuracy(y_test[ticker].values, rf_pred)
    sr_rf = sharpe_ratio(rf_pred)
    rf_mse = mean_squared_error(y_test[ticker], rf_pred)
    rf_mae = mean_absolute_error(y_test[ticker], rf_pred)
    rf_r2 = r2_score(y_test[ticker], rf_pred)

    # Store RF results
    directional_results.append([ticker, 'Random Forest', dir_acc_rf])
    sharpe_results.append([ticker, 'Random Forest', sr_rf])
    results.append([ticker, 'Random Forest', rf_mse, rf_mae, rf_r2])
    rf_models[ticker] = rf

    # --- Neural Network Hyperparameter Tuning ---
    print(f"Tuning Neural Network for {ticker}...")

    # Function to build a custom NN model
    def build_model(input_dim, neurons=64, activation='relu', learning_rate=0.001):
        model = Sequential()
        model.add(Dense(neurons, activation=activation, input_dim=input_dim))
        model.add(Dense(neurons // 2, activation=activation))
        model.add(Dense(1))  # Output layer for regression
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    # Grid of hyperparameters to try
    param_grid = {
        'neurons': [32, 64],
        'activation': ['relu', 'tanh'],
        'learning_rate': [0.001, 0.01]
    }

    best_score = float('inf')  # Initialize best score
    best_model = None
    best_params = None

    # Define local training and test sets
    X_train_local = X_train[ticker]
    y_train_local = y_train[ticker]
    X_val_local = X_test[ticker]
    y_val_local = y_test[ticker]

    # Loop through all NN combinations
    for neurons in param_grid['neurons']:
        for activation in param_grid['activation']:
            for lr in param_grid['learning_rate']:
                print(f"Training NN with neurons={neurons}, activation={activation}, lr={lr}")
                model = build_model(input_dim=X_train_local.shape[1],
                                    neurons=neurons,
                                    activation=activation,
                                    learning_rate=lr)

                # Early stopping to avoid overfitting
                es = EarlyStopping(patience=10, restore_best_weights=True)
                model.fit(X_train_local, y_train_local,
                          epochs=100,
                          verbose=0,
                          validation_split=0.2,
                          callbacks=[es])

                # Predict on validation/test data
                preds = model.predict(X_val_local)
                mse = mean_squared_error(y_val_local, preds)

                # Keep best-performing model
                if mse < best_score:
                    best_score = mse
                    best_model = model
                    best_params = {'neurons': neurons, 'activation': activation, 'learning_rate': lr}


    # Final predictions from best NN model
    nn_pred = best_model.predict(X_test[ticker])
    dir_acc_nn = directional_accuracy(y_test[ticker].values, nn_pred.flatten())
    sr_nn = sharpe_ratio(nn_pred)

    # Final metrics for NN
    nn_mse = mean_squared_error(y_test[ticker], nn_pred)
    nn_mae = mean_absolute_error(y_test[ticker], nn_pred)
    nn_r2 = r2_score(y_test[ticker], nn_pred)

    # Store results
    directional_results.append([ticker, 'Neural Network (TF)', dir_acc_nn])
    sharpe_results.append([ticker, 'Neural Network (TF)', sr_nn])
    results.append([ticker, 'Neural Network (TF)', nn_mse, nn_mae, nn_r2])
    print(f"Best NN params for {ticker}: {best_params}")
    nn_models[ticker] = best_model

# --- Compile and Display Final Results ---

# Main performance metrics: MSE, MAE, R2
results_df = pd.DataFrame(results, columns=['Stock', 'Model', 'MSE', 'MAE', 'R2'])
print("\n Model Evaluation Results ")
print(results_df.sort_values(by=['Stock', 'MSE']))

# Directional Accuracy DataFrame
dir_acc_df = pd.DataFrame(directional_results, columns=['Stock', 'Model', 'Directional Accuracy'])
print("\n Directional Accuracy ")
print(dir_acc_df.sort_values(by=['Stock', 'Directional Accuracy'], ascending=False))

# Sharpe Ratio DataFrame
sharpe_df = pd.DataFrame(sharpe_results, columns=['Stock', 'Model', 'Sharpe Ratio'])
print("\n Sharpe Ratios (Predicted Returns) ")
print(sharpe_df.sort_values(by=['Stock', 'Sharpe Ratio'], ascending=False))

# Collect Feature Importances for All Stocks
feature_importance_df = pd.DataFrame()  # Initialize an empty DataFrame to store feature importance values

# Loop through each stock ticker
for ticker in tickers:
    # Train a fresh Random Forest model on that stock's training data
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train[ticker], y_train[ticker])  # Fit RF on scaled features

    # Perform feature selection again using RFE to identify top 5 features
    estimator = LinearRegression()  # Base model for RFE
    selector = RFE(estimator, n_features_to_select=5, step=1)

    # Standardize original features (without the target column) before applying RFE
    standardized_features = StandardScaler().fit_transform(stock_data[ticker].drop(columns=['return']))
    target = stock_data[ticker]['return']

    # Fit RFE to identify most important features
    selector.fit(standardized_features, target)

    # Get names of selected features based on RFE support mask
    selected_features = stock_data[ticker].drop(columns=['return']).columns[selector.support_]

    # These are the feature names and their importances from Random Forest
    feature_names = selected_features
    importances = rf.feature_importances_  # This returns importances for the input features (i.e., the selected 5)

    # Create a temporary DataFrame for this stock's feature importances
    temp_df = pd.DataFrame({
        'Feature': feature_names,  # Names of selected features
        'Importance': importances,  # Corresponding importance scores from RF
        'Stock': ticker  # Stock ticker for context
    })

    # Concatenate this temp_df into the main feature_importance_df
    feature_importance_df = pd.concat([feature_importance_df, temp_df], axis=0)

# Now, feature_importance_df contains feature importances for all selected features across all stocks

# Plot All Feature Importances Together
plt.figure(figsize=(14, 7))  # Set figure size
# Create barplot to show feature importances across all stocks using Random Forest
sns.barplot(data=feature_importance_df, x='Feature', y='Importance', hue='Stock')
plt.title("Feature Importance Across All Stocks (Random Forest)")  # Chart title
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.tight_layout()  # Auto-adjust subplot parameters for clean layout
plt.grid(True)  # Show grid
plt.legend(title='Stock')  # Legend with title

# Actual vs Predicted Returns Plot

stock_to_plot = 'AAPL'  # Select stock to visualize actual vs predicted returns

plt.figure(figsize=(12, 5))  # Create wide figure with two subplots

# Actual vs Linear Regression Predictions
plt.subplot(1, 2, 1)  # First subplot
plt.plot(y_test[stock_to_plot].values, label='Actual', alpha=0.7)  # Actual returns
plt.plot(lr_models[stock_to_plot].predict(X_test[stock_to_plot]), label='Linear Regression', alpha=0.7)  # LR predictions
plt.title(f"{stock_to_plot} - Actual vs Linear Regression Predicted Returns")  # Title
plt.xlabel("Test Samples")
plt.ylabel("Return")
plt.legend()  # Add legend
plt.grid(True)

# Actual vs Random Forest
plt.subplot(1, 2, 2)  # Second subplot
plt.plot(y_test[stock_to_plot].values, label='Actual', alpha=0.7)  # Actual returns
plt.plot(rf_models[stock_to_plot].predict(X_test[stock_to_plot]), label='Random Forest', alpha=0.7)  # RF predictions
plt.title(f"{stock_to_plot} - Actual vs RF Predicted Returns")
plt.xlabel("Test Samples")
plt.ylabel("Return")
plt.legend()
plt.grid(True)
plt.tight_layout()  # Adjust layout


# Residual Plot (Random Forest)
stock_to_plot = 'AAPL'
# Calculate residuals: difference between actual and predicted values
rf_residuals = y_test[stock_to_plot] - rf_models[stock_to_plot].predict(X_test[stock_to_plot])

plt.figure(figsize=(8, 5))
# Plot histogram of residuals with KDE (kernel density estimate)
sns.histplot(rf_residuals, kde=True, bins=30)
plt.title(f"{stock_to_plot} - Residual Distribution (Random Forest)")
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()

# Barplot of Model Performance by Stock (R²)
plt.figure(figsize=(12, 6))
# Plot R² scores for all models grouped by stock
sns.barplot(data=results_df, x='Stock', y='R2', hue='Model')
plt.title("R² Scores of All Models by Stock")
plt.ylabel("R² Score")
plt.grid(True)
plt.legend(title='Model')
plt.tight_layout()

# Barplot of MAE (Mean Absolute Error)
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='Model', y='MAE', hue='Stock')
plt.title("MAE of Models by Stock")
plt.grid(True)
plt.tight_layout()

# Barplot of MSE (Mean Squared Error)
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x='Stock', y='MSE', hue='Model')
plt.title("MSE of Models by Stock")
plt.ylabel("Mean Squared Error (MSE)")
plt.xlabel("Stock")
plt.grid(True)
plt.legend(title='Model')
plt.tight_layout()

# Barplot of Directional Accuracy
plt.figure(figsize=(10, 6))
sns.barplot(data=dir_acc_df, x='Stock', y='Directional Accuracy', hue='Model')
plt.title("Directional Accuracy by Stock and Model")
plt.ylim(0, 1)  # Limit y-axis from 0 to 1 (since it's a proportion)
plt.grid(True)
plt.tight_layout()

# Barplot of Sharpe Ratio
plt.figure(figsize=(10, 6))
sns.barplot(data=sharpe_df, x='Stock', y='Sharpe Ratio', hue='Model')
plt.title("Sharpe Ratio of Predicted Returns by Model")
plt.grid(True)
plt.tight_layout()

# Re-run everything you want to log below

# Print the full DataFrame (or use .tail(), .head(), or .describe() for brevity)
print(" Final Dataset ")
print(df)

# Display model evaluation results (sorted by Stock and MSE)
print("\n Final Model Evaluation (MSE, MAE, R²) ")
print(results_df.sort_values(by=['Stock', 'MSE']))

# Show the cross-validation summary table for all models and stocks
print("\n Cross-Validation Summary ")
print(cv_results_df.sort_values(by=['Stock', 'MSE (CV Mean)']))

# --- Directional Accuracy ---
# Create a DataFrame from previously saved directional accuracy results
dir_acc_df = pd.DataFrame(directional_results, columns=['Stock', 'Model', 'Directional Accuracy'])

# Print directional accuracy sorted by stock and accuracy
print("\n Directional Accuracy ")
print(dir_acc_df.sort_values(by=['Stock', 'Directional Accuracy'], ascending=False))

# --- Sharpe Ratio ---
# Create DataFrame from Sharpe ratio results for all model predictions
sharpe_df = pd.DataFrame(sharpe_results, columns=['Stock', 'Model', 'Sharpe Ratio'])

# Print Sharpe ratios sorted by stock and ratio
print("\n Sharpe Ratios (Predicted Returns) ")
print(sharpe_df.sort_values(by=['Stock', 'Sharpe Ratio'], ascending=False))

# --- Feature Importances per Stock ---
print("\n Feature Importances (Random Forest) ")
for stock in tickers:
    print(f"\n--- {stock} ---")
    # Print sorted feature importances for the current stock
    print(feature_importance_df[feature_importance_df['Stock'] == stock].sort_values(by='Importance', ascending=False))

    # --- Selected Features via RFE ---
    print("\n Selected Features via RFE ")
    for ticker in tickers:
        estimator = LinearRegression()
        selector = RFE(estimator, n_features_to_select=5, step=1)

        # Get original (unscaled) features and target
        original_X = stock_data[ticker].drop(columns=['return'])
        original_y = stock_data[ticker]['return']

        # Fit RFE selector and extract selected feature names
        selector = selector.fit(StandardScaler().fit_transform(original_X), original_y)
        selected_features = original_X.columns[selector.support_]
        print(f"{ticker}: {list(selected_features)}")

# --- Number of Features Used After Selection ---
print("\n Feature Selection ")
for ticker in tickers:
    print(f"{ticker} Selected Features: {X_train[ticker].shape[1]} features used")

# --- Best SVM Parameters Found via Grid Search ---
print("\n Best SVM Parameters ")
for ticker, model in svm_models.items():
    print(f"{ticker} - SVM: {model.get_params()}")

# --- Neural Network Architecture Summary ---
print("\n Best Neural Network Configurations ")
for ticker, model in nn_models.items():
    print(f"{ticker} - NN Summary:")
    model.summary(print_fn=lambda x: print(x))  # Print NN architecture for each stock

# --- Final Model Evaluation Recap ---
print("\n Model Evaluation Results ")
print(results_df.sort_values(by=['Stock', 'MSE']))

# --- Best Model Per Stock (based on MSE) ---
print("\n Best Model Per Stock ")
for stock in tickers:
    best = results_df[results_df['Stock'] == stock].sort_values(by='MSE').iloc[0]
    print(f"{stock}: {best['Model']} (MSE: {best['MSE']:.6f}, R²: {best['R2']:.4f})")

# Show all plots at once
plt.show()