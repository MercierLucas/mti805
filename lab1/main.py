import cv2
from webcam_reader import display_webcam


face_cascade = cv2.CascadeClassifier('xml/face_cascade.xml')
eye_cascade = cv2.CascadeClassifier('xml/eye_cascade.xml')
smile_cascade = cv2.CascadeClassifier('xml/smile_cascade.xml')

def main(): 
    display_webcam(face_detection)


def add_rectangles(img, rectangles, color=(255, 0, 0)):
    for (x, y, w, h) in rectangles:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    return img
    

def detect_intra(img, detector, start_x, start_y):
    detected = detector.detectMultiScale(img, 1.3, 5)
    inside_pos = [(x + start_x, y + start_y, w, h) for (x, y, w, h) in detected]
    return inside_pos


def face_detection(img):
    # improvements: delete boxes that just appeared
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        region = img[y: y + h, x: x + w]
        gray_region = gray[y: y + h, x: x + w]
        eyes = detect_intra(gray_region, eye_cascade, x, y)
        smile = detect_intra(region, smile_cascade, x, y)
        img = add_rectangles(img, eyes, (0, 255, 0))
        img = add_rectangles(img, smile, (0, 0, 255))

    return img


def transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (9,9), 0)
    return img


def canny(img):
    return cv2.Canny(img, 100, 200)


if __name__ == '__main__':
    main()