#read and display the image
import cv2
image = cv2.imread(r"img.jpg")
#display img
cv2.imshow("first",image)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("second",gray)
#resize
resize = cv2.resize(image,(600,400))
cv2.imshow("third",resize)
#img rotation
(h,w)= image.shape[:2]
center = (w//2 , h//2)
M = cv2.getRotationMatrix2D(center,45,1.0)
rotated = cv2.warpAffine(image,M,(w,h))
cv2.imshow("fourth",rotated)
#crop
crop = image[50:200,100:300]
cv2.imshow("fifth",crop)
cv2.waitKey(0)
cv2.destroyAllWindows()