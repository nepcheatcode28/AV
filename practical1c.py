
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

#low income = 0 medium = 1, high = 2
#yes(1)buy  no(0)not buy
x= np.array([[25,0], [30,1], [45,2], [35,1], [50,0]])
y= np.array([0, 1, 1, 1, 0])
model = DecisionTreeClassifier(max_depth=3, random_state= 42)
model.fit(x,y)
prediction = model.predict([[40,2]])
print("Prediction:", prediction)
#data visualization
plt.figure(figsize= (10,6))
tree.plot_tree(model, feature_names=["Age:", "Income:"], class_names=["No", "Yes"], filled=True)
plt.show()


