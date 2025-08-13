# build a linear regression model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def linear_regression_model(data, target_column):
    """
    Build a linear regression model using the provided data.
    y = target_column
    X = all other columns in the data.
    """
    # Split the data into features and target variable
    X = data.drop(columns=[target_column])
    y = data[target_column] 
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create a linear regression model
    model = LinearRegression()      
    # Fit the model to the training data
    model.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    # Calculate the mean squared error of the predictions
    mse = mean_squared_error(y_test, y_pred)    
    print(f"Mean Squared Error: {mse}")
    return model

# call the linear regression model function with sample data containing 3 features and a target variable
if __name__ == "__main__":
    # Sample data creation
    data = pd.DataFrame({
        # 'feature1': [1, 4, 7, 10],
        'feature1': [10, 20, 30, 40],
        'feature2': [2, 5, 8, 11],
        'feature3': [3, 6, 9, 12],
        'target':   [5, 10, 15, 20]
    })
    # Call the linear regression model function
    model = linear_regression_model(data, 'target')
    print("Model coefficients:", model.coef_)
    print("Model intercept:", model.intercept_)