# Canny edge module file
import cv2

fpath = r"C:\Users\jhan7\Desktop\Anti Stable Diffusion"

def canny(imgname):
    img = cv2.imread("{fp}\original_image\{imn}.png".format(fp=fpath, imn=imgname), cv2.IMREAD_GRAYSCALE)

    min_threshold = 255/3
    max_threshold = 255
    
    edge1 = cv2.Canny(img, min_threshold, max_threshold)

    cv2.imwrite("{fp}\canny_image\{imn}_canny.png".format(fp=fpath, imn=imgname), edge1)
