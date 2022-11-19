import os, cv2

file = open("./gorani.jh", "w")

path = os.getcwd()
img = cv2.imread("{p}/images/original_image/gorani.png".format(p=path))

file.write(str(img.shape[0]) + "," + str(img.shape[1]) + "," + str(img.shape[2]) + "/")

for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        file.write(str(img[x, y][0]) + "," + str(img[x, y][1]) + "," + str(img[x, y][2]) + "/")
    file.write("ls")

file.close()