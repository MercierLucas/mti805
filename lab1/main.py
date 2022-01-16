import cv2
from argparse import ArgumentParser

from face_detector import FaceDetector
from faceid import FaceID
from face_tracking import FaceTracking
from utils import show_image, add_rectangles, Dataset
from webcam_reader import display_webcam


def train(args):
    faceid = FaceID(args['weights_dir'])
    dataset = Dataset(args['dataset'])
    faceid.train(dataset)
    faceid.save(args['pickle_dir'])


def recognize(args):
    detector = FaceDetector(args['weights_dir'])

    dataset = Dataset(args['dataset'])
    image, label_real = dataset[0]

    faces, pos = detector.face_detection(img=image, return_pos=True)
    if len(faces) == 0:
        print('No faces detected, skipping id')
        return

    faceid = FaceID(args['weights_dir'], args['pickle_dir'])

    prob, label_pred = faceid.recognize(image)
    print(f'Pred: {label_pred} {prob} | Real: {label_real}')

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    prob *= 100
    img = add_rectangles(img, pos, label=f'{label_pred}: {prob:.0f}%')
    show_image(img)


def get_parsed_args() -> ArgumentParser:
    argparser = ArgumentParser()
    argparser.add_argument("--faceid_only", dest='faceid_only', action="store_true", help="Only use faceid, or also include face detector")
    argparser.add_argument("--action", required=True, help="train or identify")
    argparser.add_argument("--dataset", help='path to input dir')
    argparser.add_argument("--pickle_dir", help='Path to pickle directory')
    argparser.add_argument("--weights_dir", help='Path to weights directory')
    argparser.add_argument("--conf", default=.5, help='conf threshold')

    return argparser
    

def main_face_detection():
    detector = FaceDetector(cascade_root_path='lab1/xml')
    display_webcam(detector.detect_and_add_shapes)


def detection_and_identification(args):
    detector = FaceDetector(args['weights_dir'])
    faceid = FaceID(args['weights_dir'], args['pickle_dir'])

    face_tracker = FaceTracking(detector, faceid)
    display_webcam(face_tracker.track)


if __name__ == '__main__':
    parser = get_parsed_args()
    args = vars(parser.parse_args())

    if args['faceid_only']:
        if args['action'] == 'train':
            train(args)
        
        elif args['action'] == 'identify':
            recognize(args)

    else:
        detection_and_identification(args)
    



