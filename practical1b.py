
import numpy as np
from sklearn.linear_model import LogisticRegression
x= np.array([[120,45], [85,30], [150,50],[70,25], [95,35], [180,55]])
# 1 = diabetic and 0 = non-diabetic
y= np.array([1,0,1,0,0,1])
#train model
model = LogisticRegression()
model.fit(x,y)
prediction = model.predict([[200,40]])
probability = model.predict_proba([[100,40]])
print("Prediction" ,prediction)
print("Probability:",probability)
