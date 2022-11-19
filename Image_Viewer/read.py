import numpy as np
import cv2, zlib

file = open("gorani.jh", "rb")
content = file.read()
content = zlib.decompress(content).decode("utf-8")

stmp = content[:9]
size = [int(stmp[:3]), int(stmp[3:6]), int(stmp[6:9])]
content = list(content.split(" "))

img = np.zeros((size[0], size[1], size[2]), dtype=np.uint8)

for i in range(size[0]):
    for j in range(1, size[1]):
        sp = j*9
        img[i][j] = [int(content[i][sp:sp+3]), int(content[i][sp+3:sp+6]), int(content[i][sp+6:sp+9])]

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

file.close()
