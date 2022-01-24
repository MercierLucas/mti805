from re import A
import cv2
from argparse import ArgumentParser
from sklearn.metrics import classification_report, confusion_matrix

from face_detector import FaceDetector
from faceid import FaceID
from face_tracking import FaceTracking
from utils import show_image, Dataset, load_image, show_classification_results, add_border
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
    #argparser.add_argument("--add_detector", dest='add_detector', action="store_true", help="Add face detector or not, if not faceid on the entire image")
    argparser.add_argument("--action", required=True, help="Action to perform like train, identify etc.")
    argparser.add_argument("--input", required=True, help="'webcam' or a path to an image or a folder")
    argparser.add_argument("--pickle_dir", help='Path to pickle directory')
    argparser.add_argument("--weights_dir", help='Path to weights directory')

    return argparser
    

def face_detection(args):
    detector = FaceDetector(cascade_root_path=args['weights_dir'])
    if args['input'] == 'webcam':
        display_webcam(detector.detect_and_add_shapes)
    else:
        image = load_image(args['input'])
        image = detector.detect_and_add_shapes(image)
        show_image(image)
    


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


def evaluate(args):
    dataset = Dataset(args['input'], recursive=True)
    faceid = FaceID(args['weights_dir'], args['pickle_dir'])
    labels_pred = []
    labels_true = []
    results = {}
    for image, label in dataset:
        _, label_pred = faceid.recognize(image)
        labels_pred.append(label_pred)
        labels_true.append(label)
        if label_pred not in results:
            results[label_pred] = []

        image = add_border(image, label_pred == label)
        
        results[label_pred].append(image)

    print("Confusion matrix:")
    show_classification_results(results)
    print(confusion_matrix(labels_true, labels_pred))
    print("Classification report")
    print(classification_report(labels_true, labels_pred))



if __name__ == '__main__':
    parser = get_parsed_args()
    args = vars(parser.parse_args())

    actions = {
        'face_detection': face_detection,
        'evaluate': evaluate,
        'train': train,
        'detect_and_id': detection_and_identification 
    }

    assert args['action'] in actions, f'{args["action"]} not available, use one of: {actions.keys()}'

    actions[args['action']](args)
    exit()

    



