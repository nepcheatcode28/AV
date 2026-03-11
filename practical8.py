import cv2 
from ultralytics import YOLO 
import matplotlib.pyplot as plt 
model = YOLO("yolov8n.pt") 
image_path=r"C:drama.png"
img=cv2.imread(image_path) 
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
result=model(img_rgb) 
annoted_img = result[0].plot()
cv2.imshow("Detection",annoted_img)
plt.imshow(cv2.cvtColor(annoted_img,cv2.COLOR_BGR2RGB))
plt.axis("Off")
plt.show()
cv2.waitKey(0)
