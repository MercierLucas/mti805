import cv2
from os import listdir
from os.path import isfile, join


def list_files_in_dir_recursive(path):
    """Load all all images in subdirs"""
    subfolders = list(listdir(path))
    images = []
    for folder in subfolders:
        sub_path = join(path, folder)
        for f in listdir(sub_path):
            file_path = join(sub_path, f)
            if isfile(file_path):
                images.append(file_path)
    return images


def load_image(img_path):
    file_ext = img_path.split('.')[-1]
    assert file_ext in ['png', 'jpg', 'jpeg'], 'The input must be an image'

    image = cv2.imread(img_path)
    assert image is not None, f'Error while loading the image {img_path}'

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

