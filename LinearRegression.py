from sklearn.linear_model import LinearRegression
import numpy as np

# Input data (study hours)
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)

# Output data (marks)
y = np.array([10, 20, 30, 40, 50, 60])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict marks for 7 study hours
prediction = model.predict([[7]])

print("Predicted Marks:", prediction[0])

# Show slope and intercept
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
