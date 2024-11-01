# build_dataset.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load the dataset from a CSV file and split it into training and testing sets."""
    data = pd.read_csv(file_path)
    X = data.drop(columns=['species'])  # Assuming the target column is named 'species'
    y = data['species']

    # Encoding the target variable
    y = pd.factorize(y)[0]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data("iris.csv")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
