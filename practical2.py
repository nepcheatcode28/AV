
import numpy as np
from sklearn.linear_model import LogisticRegression
x = np.array([[2], [4], [6], [8], [10]]) #study hours
y = np.array([0,0,1,1,1])
model = LogisticRegression()
model.fit(x,y)
prediction = model.predict([[5]])
probability = model.predict_proba([[5]])
print("Prediction:", prediction)
print("Probability:",probability)
