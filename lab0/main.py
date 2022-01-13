import cv2
from cv_utils.entities.image import Image
from cv_utils.visualizer.image_viz import show_images, show_image



def hist():
    img = Image('peppers.png')
    #show_rgb_hist(img.rgb)


def sobel_filter(img):
    grad_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad



def filters():
    img = Image('coins.png').grayscale
    sobel = sobel_filter(img)
    canny = cv2.Canny(sobel, 150, 200)
    show_images([img, canny], ['Base', 'Canny'])


def hsv():
    img = Image('peppers.png')
    img = cv2.cvtColor(img.image ,cv2.COLOR_BGR2HSV)
    show_image(img)


if __name__ == '__main__':
    hsv()