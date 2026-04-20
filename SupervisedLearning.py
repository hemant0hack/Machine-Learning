import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5]])

y = np.array([30, 40, 50, 60, 70])

model = LinearRegression()

model.fit(X, y)

hours = [[6]]
prediction = model.predict(hours)

print("Predicted Marks:", prediction)
