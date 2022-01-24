import cv2
import matplotlib.pyplot as plt
import numpy as np


def add_rectangles(img, rectangles, color=(255, 0, 0), label=None, thickness=None):
    if color == 'random':
        color = list(np.random.random(size=3) * 256)

    if thickness is None:
        size, thickness = _compute_text_misc(img)
    else:
        size, _ = _compute_text_misc(img)
        
    for (x, y, w, h) in rectangles:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness*2)
        if label is not None:
            img = cv2.putText(img, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, 2)
    return img


def add_border(img, is_correct):
    color = (0, 255, 0) if is_correct else (255, 0, 0)
    border = [10, 10, img.shape[1] - 20, img.shape[0] - 20]
    res = add_rectangles(img, [border], color=color, thickness=10)
    return res


def _compute_text_misc(img):
    h, w = img.shape[:2]
    font_size = min(h, w) * 0.0025      # ratio from tests
    thickness = 2
    return font_size, thickness


def show_classification_results(results):
    plt.figure(figsize=(5,5))
    rows = len(results.keys())
    cols = max(len(imgs) for _, imgs in results.items())
    for y, (label, images) in enumerate(results.items()):
        for x, image in enumerate(images):
            plt.subplot(rows, cols, y*cols + (x + 1))
            plt.axis('off')
            plt.title(label)
            plt.imshow(image)
    plt.show()



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

    plt.axis('off')
    plt.show()