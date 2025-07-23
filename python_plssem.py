import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# Load the data
latent_scores = pd.read_csv('../data/latent_scores.csv')  # Latent variable scores
data_test_model = pd.read_csv('../data/temp_for_q.csv')  # Original dataset

# Assume 'data' is your original dataset
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_test_model)  # Scale all columns in the dataset

# Convert the scaled data back to a DataFrame for easy manipulation
data_scaled_df = pd.DataFrame(data_scaled, columns=data_test_model.columns)

# Prepare the data
# Extract only the relevant columns for the path model
X = data_scaled_df[['WR', 'CSSD', 'BWBp', 'Price', 'Sentiment', 'InvestorAttention']]  # Indicators
Y = data_scaled_df[['TradingVolume', 'TurnoverRatio']]  # Dependent variable for Q² calculation (CSSD)

# Add the latent variable scores to X (latent scores as features)
X = pd.concat([X, latent_scores], axis=1)

# Extract latent variables as individual components
herding = latent_scores['Herding'] * 0.112
overconfidence = latent_scores['Overconfidence'] * -0.436
fomo = latent_scores['FoMO'] * 0.662
market_instability = latent_scores['MarketInstability']

# Set the path model equations:
# 1. Herding = -0.127 * CSSD + 0.127 * WR (path)
# 2. Overconfidence = 0.882 * TradingVolume + 0.882 * TurnoverRatio (path)
# 3. FoMO = -0.226 * SentimentAdj (path)
# 4. MarketInstability = ... (We have the latent score here)

# Train a regression model to predict CSSD (as an example using Ridge Regression)
from sklearn.linear_model import Ridge

# Let's first predict the CSSD variable (dependent variable)
model = Ridge(alpha=1.0)  # Ridge Regression for simplicity

# Train the model with the selected features
model.fit(X, Y)

# Predict CSSD values
predicted_cssd = model.predict(X)

# Calculate Q² for prediction quality
# Residual Sum of Squares (RSS)
rss = np.sum((Y.values - predicted_cssd) ** 2)

# Total Sum of Squares (TSS)
tss = np.sum((Y.values - np.mean(Y.values)) ** 2)

# Q² Calculation
q2_value = 1 - (rss / tss)

print(f"Q² value: {q2_value}")

