# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Sample dataset (Hours studied vs Pass(1)/Fail(0))
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Create Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X, y)

# Predict for new data
hours = [[4.5]]
prediction = model.predict(hours)

print("Prediction (0 = Fail, 1 = Pass):", prediction)

# Plot data
plt.scatter(X, y, color='blue')
plt.xlabel("Study Hours")
plt.ylabel("Pass / Fail")
plt.title("Logistic Regression Example")
plt.show()