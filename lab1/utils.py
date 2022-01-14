import cv2

def add_rectangles(img, rectangles, color=(255, 0, 0)):
    for (x, y, w, h) in rectangles:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    return img

