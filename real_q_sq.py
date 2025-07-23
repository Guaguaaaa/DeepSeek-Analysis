import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Load the data
latent_scores = pd.read_csv('../data/latent_variable_scores +30.csv')  # Latent variable scores
data_test_model = pd.read_csv('../data/final_data/post_30_data.csv')  # Original dataset

data_test_model = data_test_model.drop(columns=['Date', 'Stocks'])

# Scale the original data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_test_model)
data_scaled_df = pd.DataFrame(data_scaled, columns=data_test_model.columns)

# Prepare the data
X_indicators = data_scaled_df[['CSSD', 'BWBp', 'Price', 'Sentiment', 'InvestorAttention']]
Y_endogenous = data_scaled_df[['TradingVolume', 'TurnoverRatio']]

# Add latent scores to X
X = pd.concat([X_indicators, latent_scores], axis=1)

# Path model equations (replace with your actual calculated path coefficients)
# These are example path coefficients. You MUST replace them with your actual model's results.
path_coefficients = {
    'Herding': {'CSSD': 0.067},
    'Overconfidence': {'BWBp': -0.495, 'Price': -0.495},
    'FoMO': {'Sentiment': 0.701, 'InvestorAttention': 0.701},
    # MarketInstability is already in latent scores, so no path needed
}

# Function to calculate predicted latent scores (based on path model)
def predict_latent_scores(data, path_coefs):
    predicted_latent = pd.DataFrame(index=data.index)
    for latent, coefs in path_coefs.items():
        predicted_latent[latent] = 0
        for indicator, coef in coefs.items():
            predicted_latent[latent] += data[indicator] * coef
    return predicted_latent

# Function to calculate Q² using blindfolding (K-fold cross-validation)
def calculate_q2(X_indicators, Y_endogenous, latent_scores, path_coefficients, n_folds=10):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    q2_values = {}

    for train_index, test_index in kf.split(X_indicators):
        X_train, X_test = X_indicators.iloc[train_index], X_indicators.iloc[test_index]
        Y_train, Y_test = Y_endogenous.iloc[train_index], Y_endogenous.iloc[test_index]
        latent_train = latent_scores.iloc[train_index]
        latent_test = latent_scores.iloc[test_index]

        # Predict latent scores based on the training data's path model
        predicted_latent_train = predict_latent_scores(X_train, path_coefficients)

        # Train a regression model to predict endogenous variables
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(pd.concat([X_train, predicted_latent_train, latent_train], axis=1), Y_train)

        # Predict endogenous variables on the test set
        predicted_endogenous = model.predict(pd.concat([X_test, predict_latent_scores(X_test, path_coefficients), latent_test], axis=1))

        # Calculate Q² for each endogenous variable
        for col in Y_endogenous.columns:
            rss = np.sum((Y_test[col].values - predicted_endogenous[:, Y_endogenous.columns.get_loc(col)]) ** 2)
            tss = np.sum((Y_test[col].values - np.mean(Y_test[col].values)) ** 2)
            q2 = 1 - (rss / tss)
            if col not in q2_values:
                q2_values[col] = []
            q2_values[col].append(q2)

    # Average Q² values across folds
    average_q2 = {col: np.mean(values) for col, values in q2_values.items()}
    return average_q2


def calculate_weighted_q2(q2_values, loadings):
    """
    Calculates a weighted average of Q² values based on loadings.

    Args:
        q2_values (dict): A dictionary where keys are indicator names and values are Q² values.
        loadings (dict): A dictionary where keys are indicator names and values are loadings.

    Returns:
        float: The weighted average Q².
    """

    # Ensure that both dictionaries have the same keys
    if q2_values.keys() != loadings.keys():
        raise ValueError("Q² values and loadings dictionaries must have the same keys.")

    # Calculate weighted Q²
    weighted_sum = 0
    total_weight = 0

    for indicator, q2 in q2_values.items():
        loading = loadings[indicator]
        weighted_sum += q2 * loading
        total_weight += loading

    if total_weight == 0:
        return 0  # Avoid division by zero

    weighted_q2 = weighted_sum / total_weight
    return weighted_q2


# Calculate and print Q² values
q2_results = calculate_q2(X_indicators, Y_endogenous, latent_scores, path_coefficients)
loadings = {'TradingVolume': 0.861, 'TurnoverRatio': 0.733} #Replace with your actual loadings.
print("Q² values:", q2_results)

weighted_q2_overconfidence = calculate_weighted_q2(q2_results, loadings)
print(f"Weighted Q² for Overconfidence: {weighted_q2_overconfidence}")