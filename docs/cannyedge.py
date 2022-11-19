import cv2
from matplotlib import pyplot as plt

img = cv2.imread('./egon.png',0)
edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

# https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html