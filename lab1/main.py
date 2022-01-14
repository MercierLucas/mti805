import cv2
import argparse
from webcam_reader import display_webcam
from face_detection import FaceDetector



def main_face_detection():
    detector = FaceDetector(cascade_root_path='lab1/xml')
    display_webcam(detector.face_detection)


if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--action', default='detection', help='Detection only')
    main_face_detection()