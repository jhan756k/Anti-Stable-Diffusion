import cv2, os
import numpy as np
from cannyEdge import canny

'''
kernel2d = np.array([
    [0.9782436, 0.97248424, 0.97034105, 0.97248424, 0.9782436 ],
    [0.97248424, 0.98568037, 0.98314849, 0.98568037, 0.97248424],
    [0.97034105, 0.98314849, 0.99047197, 0.98314849, 0.97034105],
    [0.97248424, 0.98568037, 0.98314849, 0.98568037, 0.97248424],
    [0.9782436, 0.97248424, 0.97034105, 0.97248424, 0.9782436 ]
])
kernel2d = np.array(kernel2d)
'''

path = os.getcwd()
imgname = "forest"
canny(imgname)

cannyimg = cv2.imread("{p}/images/canny_image/{ing}_canny.png".format(p=path, ing=imgname), cv2.IMREAD_GRAYSCALE)
orig_img = cv2.imread("{p}/images/original_image/{ing}.png".format(p=path, ing=imgname))

n = 3 # 커널 크기

for x in range(n-2, cannyimg.shape[0]-(n-2)):
    for y in range(n-2, cannyimg.shape[1]-(n-2)):
        if cannyimg[x][y] == 255:
            
            for p in range(3):
                sum = 0

                for i in range(n):
                    for j in range(n):
                        sum += orig_img[x+i-(n-2)][y+j-(n-2)][p] # 커널 만들어서 여따 곱해도 됨 근데 너무 느림
                orig_img[x][y][p] = sum // 9

            '''
            for tx in range(5):
                for ty in range(5):
                    
                    for i in range(3):
                        if 0<=x+tx-2<cannyimg.shape[0] and 0<=y+ty-2<cannyimg.shape[1]:
                            orig_img[x+tx-2][y+ty-2][i] *= kernel2d[tx][ty]
            '''

cv2.imwrite("{p}/images/edgeblur_image/{ing}_edgeblur.png".format(p=path, ing=imgname), orig_img)

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