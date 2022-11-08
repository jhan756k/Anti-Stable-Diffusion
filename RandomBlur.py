import cv2
import random
image = cv2.imread("./images/monalisa.png")

blur_pixel = 3000

for _ in range(blur_pixel):
    x = random.randint(0, image.shape[0] - 1)
    y = random.randint(0, image.shape[1] - 1)
    image[x, y] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

cv2.imwrite("./images/monalisa_blur.png", image)
