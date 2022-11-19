import cv2, random
image = cv2.imread("./images/original_image/egon.png")

blur_rate = 1
cnt = 0

for x in range(image.shape[0]):
    for y in range(image.shape[1]):
        if random.random() < blur_rate:
            image[x, y] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cnt+=1

print(cnt/(image.shape[0]*image.shape[1])*100)

cv2.imwrite("./images/randomblur_image/egon_blur.png", image)
