import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def calculate_vif(df):
    """
    Calculates the Variance Inflation Factor (VIF) for each variable in a DataFrame.

    Args:
        df: pandas DataFrame.  Must contain only the *numeric* independent variables.
             Categorical variables MUST be pre-processed.

    Returns:
        pandas DataFrame: VIF values for each variable.
    """
    # Add a constant term to the DataFrame for the intercept
    X = add_constant(df)

    # Drop the constant-value-column if there is one
    for col in X.columns:
        if len(X[col].unique()) == 1:
            X = X.drop(columns=col)
            print(f"Warning: Dropped constant column '{col}' before VIF calculation.")

    # Calculate VIF for each variable
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data


def analyze_multicollinearity(file_path):
    """
    Analyzes multicollinearity in a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        None.  Prints the correlation matrix, displays a heatmap, and prints VIF values.
        Prints error messages if issues are encountered.
    """
    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)

        # Drop rows with any missing values (NaN)
        df = df.dropna()

        # Remove any completely empty columns
        df = df.dropna(axis=1, how='all')

        # Select only numeric columns for correlation and VIF calculation
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.shape[1] < 2:
            print(
                "Error: Less than two numeric columns after data cleaning.  Cannot perform multicollinearity analysis.")
            return

        # Calculate the correlation matrix
        correlation_matrix = numeric_df.corr()

        # Print the correlation matrix
        print("Correlation Matrix:")
        print(correlation_matrix)

        # Visualize the correlation matrix using a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix Heatmap")
        plt.show()

        # Calculate VIF
        vif_df = calculate_vif(numeric_df)

        # Print VIF values
        print("\nVariance Inflation Factors (VIF):")
        print(vif_df)


    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Example usage: Replace 'testy.csv' with the actual path to your file
file_path = 'data/testy.csv'
analyze_multicollinearity(file_path)