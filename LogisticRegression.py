import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Input: study hours
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)

# Output: 0 = fail, 1 = pass
y = np.array([0, 0, 0, 1, 1, 1])

# Create model
model = LogisticRegression()

# Train model
model.fit(X, y)

# Predict for 5 hours
prediction = model.predict([[5]])
probability = model.predict_proba([[5]])

print("Prediction:", prediction[0])
print("Probability:", probability)

plt.scatter(X, y, color='blue')
plt.xlabel("Study Hours")
plt.ylabel("Pass / Fail")
plt.title("Logistic Regression Example")
plt.show()
