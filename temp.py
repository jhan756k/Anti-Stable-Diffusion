import cv2, os
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from cannyEdge import canny

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

cv2.imwrite("{p}/images/edgeblur_image/{ing}_edgeblur.png".format(p=path, ing=imgname), orig_img)
print(imgname, psnr(cv2.imread("{p}/images/original_image/{ing}.png".format(p=path, ing=imgname)), cv2.imread("{p}/images/edgeblur_image/{ing}_edgeblur.png".format(p=path, ing=imgname))), ssim(cv2.imread("{p}/images/original_image/{ing}.png".format(p=path, ing=imgname)), cv2.imread("{p}/images/edgeblur_image/{ing}_edgeblur.png".format(p=path, ing=imgname)), multichannel=True))
