import numpy as np
import cv2
import os
import glob
from skimage.measure import compare_psnr

orig_images = []
for ii in range(90,100):
	n = cv2.imread("data2017/etc/test/img_%d.png" % ii,0)
	orig_images.append(n)
# print(orig_images[0].shape)

btc_images = []
for ii in range(0,10):
	n = cv2.imread("samples_btc_train3_evaluate/evaluate_bsd100/valid_lr_%d.png" % ii,0)
	btc_images.append(n)

gen_images = []
for ii in range(0,10):
	n = cv2.imread("samples_btc_train3_evaluate/evaluate_bsd100/valid_gen_%d.png" % ii,0)
	gen_images.append(n)

# # print(orig_images[0].shape)
print("generated images.")
psnr = 0
for i in range(0,10):
	psnr = psnr + compare_psnr(orig_images[i],gen_images[i])

psnr = psnr / 10 
print("psnr generated: ", psnr)

# for i in range(0,10):
# 	print(i, ": ",compare_psnr(orig_images[i],gen_images[i]))



print("btc images.")
psnr = 0
for i in range(0,10):
	psnr = psnr + compare_psnr(orig_images[i],btc_images[i])

psnr = psnr / 10 
print("psnr odbtc: ", psnr)

# for i in range(0,10):
# 	print(i, ": ",compare_psnr(orig_images[i],btc_images[i]))