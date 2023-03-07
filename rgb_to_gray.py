import cv2

img_rgb = cv2.imread('IMG_9405.jpeg')

im = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

cv2.imwrite('IMG_9405_gray.jpeg', im)