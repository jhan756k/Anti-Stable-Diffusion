import numpy as np
import cv2, zlib
import time

start = time.perf_counter()

file = open("eagle.jhp", "rb")
content = zlib.decompress(file.read()).decode("utf-8")
stmp = content[:12]
size = [int(stmp[:4]), int(stmp[4:8]), int(stmp[8:12])]
content = list(content.split(" "))
img = np.zeros((size[0], size[1], size[2]), dtype=np.uint8)

sp = 12
for i in range(size[0] - 1):
    for j in range(size[1] - 1):
        img[i][j] = [int(content[i][sp:sp+3]), int(content[i][sp+3:sp+6]), int(content[i][sp+6:sp+9])]
        sp += 9
    sp=0

end = time.perf_counter()

print("Time taken: {t}".format(t=end-start))

cv2.imshow("Press c to terminate", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

file.close()