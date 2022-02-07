from argparse import ArgumentParser

from actions import Action, MatchAction, StitchAction, FeatureDetectAction


def get_parsed_args() -> ArgumentParser:
    argparser = ArgumentParser()
    argparser.add_argument("--action", required=True, help="Action to perform")
    argparser.add_argument("--input", required=True, help="Folder to process")
    argparser.add_argument("--feature_detector", required=True, help="Algorithm to use to detect keypoints")

    return vars(argparser.parse_args())
    

if __name__ == '__main__':
    actions = {
        'stitch': StitchAction,
        'detect': FeatureDetectAction,
        'match': MatchAction
    }

    args = get_parsed_args()
    assert args['action'] in actions, f'{args["action"]} not available, use one of: {actions.keys()}'

    action = actions[args['action']](args)
    action.perform()

