import os, cv2, zlib

file = open("eagle.jhp", "wb")

path = os.getcwd()
img = cv2.imread("{p}/images/research_image/eagle.jpeg".format(p=path))

text = str(img.shape[0]).zfill(4) + str(img.shape[1]).zfill(4) + str(img.shape[2]).zfill(4)

for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        text += (str(img[x, y][0]).zfill(3) + str(img[x, y][1]).zfill(3) + str(img[x, y][2]).zfill(3))
    text += " "

comp_text = zlib.compress(text.encode("utf-8"), 9)
file.write(comp_text)
file.close()