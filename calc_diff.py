from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import cv2

img1 = cv2.imread(r'C:\Users\jhan7\Desktop\Anti-Stable-Diffusion\images\original_image\eagle.png')

img2 = cv2.imread(r"C:\Users\jhan7\Desktop\Anti-Stable-Diffusion\images\edgeblur_image\eagle_edgeblur.png")

print(psnr(img1, img2))
print(ssim(img1, img2, multichannel=True))
