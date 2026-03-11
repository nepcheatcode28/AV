
#Step 1: import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#Step 2: Data to train
x = np.array([ 1000, 1500, 2000, 2500, 3000 ]).reshape(-1,1) #we call this as a feature
y = np.array([ 150000, 200000, 250000, 300000, 350000 ])
#Train the model
model = LinearRegression() #feature
model.fit(x,y) #target
print("intercept =", model.intercept_)
print("slope =", model.coef_[0])
#Predict the price for "2200"
predicted_price = model.predict([[ 2200 ]])
print("Predicted price =", predicted_price)
#Data Visualisation
plt.scatter(x, y, color = "blue", label="actual data")
plt.plot(x, model.predict(x), color = "red", label = "regression")
plt.scatter(2200, predicted_price, color="green", label = "Predicted Price")
plt.xlabel("Square Footage")
plt.ylabel("Price")
plt.legend()
plt.show()
