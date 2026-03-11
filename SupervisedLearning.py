# Import libraries
import numpy as np
from sklearn.linear_model import LinearRegression

# Training data (Input and Output)
# X = Study hours
X = np.array([[1], [2], [3], [4], [5]])

# y = Marks obtained
y = np.array([30, 40, 50, 60, 70])

# Create model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predict for new data
hours = [[6]]
prediction = model.predict(hours)

print("Predicted Marks:", prediction)