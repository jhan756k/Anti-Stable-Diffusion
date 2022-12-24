import cv2, os
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
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
imgname = "eagle"
canny(imgname)

cannyimg = cv2.imread("{p}/images/canny_image/{ing}_canny.png".format(p=path, ing=imgname), cv2.IMREAD_GRAYSCALE)   
orig_img = cv2.imread("{p}/images/original_image/{ing}.png".format(p=path, ing=imgname))

kernelSize = 5 # 블러 커널 크기 
nearbyBlurSize = 5 # 블러 처리할 주변 픽셀 수

term = (kernelSize//2) + (nearbyBlurSize//2)

for x in range(term, cannyimg.shape[0]-term):
        for y in range(term, cannyimg.shape[1]-term):
            if cannyimg[x][y] == 255:
                
                for t in range(x-(nearbyBlurSize//2), x+(nearbyBlurSize//2)):
                    for c in range(y-(nearbyBlurSize//2), y+(nearbyBlurSize//2)):
                            
                        for p in range(3):
                            sum = 0

                            for i in range(kernelSize):
                                for j in range(kernelSize):
                                    sum += orig_img[t+i-term][c+j-term][p] 
                            orig_img[t][c][p] = sum // (kernelSize*kernelSize)

            '''
            for tx in range(5):
                for ty in range(5):
                    
                    for i in range(3):
                        if 0<=x+tx-2<cannyimg.shape[0] and 0<=y+ty-2<cannyimg.shape[1]:
                            orig_img[x+tx-2][y+ty-2][i] *= kernel2d[tx][ty]
            '''

cv2.imwrite("{p}/images/edgeblur_image/{ing}_edgeblur.png".format(p=path, ing=imgname), orig_img)
print(imgname, psnr(cv2.imread("{p}/images/original_image/{ing}.png".format(p=path, ing=imgname)), cv2.imread("{p}/images/edgeblur_image/{ing}_edgeblur.png".format(p=path, ing=imgname))), ssim(cv2.imread("{p}/images/original_image/{ing}.png".format(p=path, ing=imgname)), cv2.imread("{p}/images/edgeblur_image/{ing}_edgeblur.png".format(p=path, ing=imgname)), multichannel=True))

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