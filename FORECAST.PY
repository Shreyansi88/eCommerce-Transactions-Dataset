from sklearn.model_selection import train_test_split
from sklearn.linear_regression import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Prepare features and target
X = merged_data[['Quantity', 'Price']]
y = merged_data['TotalValue']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Example prediction for a future transaction
future_transaction = pd.DataFrame({'Quantity': [2], 'Price': [100]})
predicted_total_value = model.predict(future_transaction)

print(f"Predicted total value for a transaction with Quantity=2 and Price=100: ${predicted_total_value[0]:.2f}")

print("Predictive model completed.")