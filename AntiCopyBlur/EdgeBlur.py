import cv2
from cannyEdge import canny

fpath = r"C:\Users\jhan7\Desktop\Anti Stable Diffusion"
imgname = "astro"

canny(imgname)

cannyimg = cv2.imread("{fp}\canny_image\{imn}_canny.png".format(fp=fpath, imn=imgname), cv2.IMREAD_GRAYSCALE)
orig_img = cv2.imread("{fp}\original_image\{imn}.png".format(fp=fpath, imn=imgname))

for x in range(0, cannyimg.shape[0]):
    for y in range(0, cannyimg.shape[1]):
        if cannyimg[x][y] == 255:
            # 가우스 필터 구현하고 원본이미지의 (x,y)에 가우스 커널 적용 
            # 3*3 or 5*5 kernel, sigma = 3
            pass

print(cannyimg)