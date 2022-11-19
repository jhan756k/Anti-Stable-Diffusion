import cv2
import numpy as np
from cannyEdge import canny

# 가우스 필터 구현하고 원본이미지의 (x,y)에 가우스 커널 적용 
# 5*5 kernel, sigma = 3

imgname = "forest"

kernel = cv2.getGaussianKernel(5, 3)
kernel2d = np.outer(kernel, kernel.transpose())
kernel2d-=1
kernel2d= np.abs(kernel2d)
kernel2d+=0.01
print(kernel2d)

fpath = r"C:\Users\jhan7\Desktop\Anti Stable Diffusion"

canny(imgname)

cannyimg = cv2.imread("{fp}\canny_image\{imn}_canny.png".format(fp=fpath, imn=imgname), cv2.IMREAD_GRAYSCALE)
orig_img = cv2.imread("{fp}\original_image\{imn}.png".format(fp=fpath, imn=imgname))

for x in range(0, cannyimg.shape[0]):
    for y in range(0, cannyimg.shape[1]):
        if cannyimg[x][y] == 255:
            
            for tx in range(5):
                for ty in range(5):
                    
                    for i in range(3):
                        if 0<=x+tx-2<cannyimg.shape[0] and 0<=y+ty-2<cannyimg.shape[1]:
                            orig_img[x+tx-2][y+ty-2][i] *= kernel2d[tx][ty]
            
cv2.imwrite("{fp}\edgeblur_image\{imn}_edgeblur.png".format(fp=fpath, imn=imgname), orig_img)

'''
dx = [1, 0, -1, 0]
dy = [0, 1, 0, -1]

for it in range(3):
                orig_img[x][y][it]*=0.8

            for i in range(4):
                tmp_x = x + dx[i]
                tmp_y = y + dy[i]
                if tmp_x>0 and tmp_x<cannyimg.shape[0] and tmp_y>0 and tmp_y<cannyimg.shape[1]:
                    for it in range(3):
                        orig_img[tmp_x][tmp_y][it]*=0.9
'''
