import cv2, random
image = cv2.imread("./images/research_image/eagle.jpeg")

blur_rate = 1
cnt = 0

for x in range(image.shape[0]):
    for y in range(image.shape[1]):
        if random.random() < blur_rate:
            image[x, y] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cnt+=1

print(cnt/(image.shape[0]*image.shape[1])*100)
# 10 25 80 100
cv2.imwrite("./images/randomblur_image/eagle_blur_{r}.png".format(r=blur_rate), image)
