import os, cv2

file = open("./gorani.jh", "w")

path = os.getcwd()
img = cv2.imread("{p}/images/original_image/gorani.png".format(p=path))

file.write(str(img.shape[0]).zfill(3) + str(img.shape[1]).zfill(3) + str(img.shape[2]).zfill(3))

for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        file.write(str(img[x, y][0]).zfill(3) + str(img[x, y][1]).zfill(3) + str(img[x, y][2]).zfill(3))
    file.write(" ")

file.close()