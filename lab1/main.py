import cv2
from argparse import ArgumentParser

from face_detector import FaceDetector
from faceid import FaceID
from face_tracking import FaceTracking
from utils import show_image, Dataset, load_image
from webcam_reader import display_webcam


def train(args):
    faceid = FaceID(args['weights_dir'])
    dataset = Dataset(args['dataset'])
    faceid.train(dataset, save_path=args['pickle_dir'])


def recognize(args):
    detector = FaceDetector(args['weights_dir'])
    faceid = FaceID(args['weights_dir'], args['pickle_dir'])
    face_tracker = FaceTracking(detector, faceid)

    image = load_image(args['input'])

    image = face_tracker.track(image, verbose=True)
    show_image(image)


def get_parsed_args() -> ArgumentParser:
    argparser = ArgumentParser()
    argparser.add_argument("--add_detector", dest='add_detector', action="store_true", help="Add face detector or not, if not faceid on the entire image")
    argparser.add_argument("--faceid_action", required=True, help="train or identify")
    argparser.add_argument("--input", required=True, help="'webcam' or a path to an image")
    argparser.add_argument("--dataset", help='Path to input dir')
    argparser.add_argument("--pickle_dir", help='Path to pickle directory')
    argparser.add_argument("--weights_dir", help='Path to weights directory')

    return argparser
    

def main_face_detection():
    detector = FaceDetector(cascade_root_path='lab1/xml')
    display_webcam(detector.detect_and_add_shapes)


def detection_and_identification(args):
    detector = FaceDetector(args['weights_dir'])
    faceid = FaceID(args['weights_dir'], args['pickle_dir'])

    face_tracker = FaceTracking(detector, faceid)

    if args['input'] == 'webcam':
        display_webcam(face_tracker.track)
    else:
        image = load_image(args['input'])
        image = face_tracker.track(image)
        show_image(image)



if __name__ == '__main__':
    parser = get_parsed_args()
    args = vars(parser.parse_args())

    if not args['add_detector']:
        assert args['input'] != 'webcam', 'Webcam can\'t be used without face detector'

        if args['faceid_action'] == 'train':
            train(args)
        
        elif args['faceid_action'] == 'identify':
            recognize(args)

    else:
        detection_and_identification(args)
    



