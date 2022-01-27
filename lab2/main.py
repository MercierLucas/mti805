from argparse import ArgumentParser

from actions import Action, MatchAction, StitchAction


def get_parsed_args() -> ArgumentParser:
    argparser = ArgumentParser()
    argparser.add_argument("--action", required=True, help="Action to perform")
    argparser.add_argument("--input", required=True, help="Folder to process")

    return vars(argparser.parse_args())


#def feature_detection(args):
#    detector = FeatureDetector('sift')
#    image = load_image(args['input'])
#    show_image(detector.get_keypoints(image))
    

if __name__ == '__main__':
    actions = {
        'stitch': StitchAction,
        #'feature_detection': feature_detection,
        'match': MatchAction
    }

    args = get_parsed_args()
    assert args['action'] in actions, f'{args["action"]} not available, use one of: {actions.keys()}'

    action = actions[args['action']](args)
    action.perform()

