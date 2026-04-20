import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

model = LogisticRegression()

model.fit(X, y)

hours = [[4.5]]
prediction = model.predict(hours)

print("Prediction (0 = Fail, 1 = Pass):", prediction)

plt.scatter(X, y, color='blue')
plt.xlabel("Study Hours")
plt.ylabel("Pass / Fail")
plt.title("Logistic Regression Example")
plt.show()
