import cv2
import matplotlib.pyplot as plt
import numpy as np


def add_rectangles(img, rectangles, color=(255, 0, 0), label=None):
    if color == 'random':
        color = list(np.random.random(size=3) * 256)
        
    for (x, y, w, h) in rectangles:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), color, 10)
        if label is not None:
            img = cv2.putText(img, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 5, 2)
    return img




def show_image(img, title='', to_rgb=False, size=None):
    """Show a single image"""
    if size:
        plt.figure(figsize=(size,size))
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    if title != '':
        plt.title(title)
    plt.show()