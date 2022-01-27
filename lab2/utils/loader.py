import cv2
from os import listdir
from os.path import isfile, join


def list_files(path):
    images = []
    elements = list(listdir(path))
    for el in elements:
        full_path = join(path, el)
        if isfile(full_path):
            images.append(full_path)
    return images


def load_image(img_path):
    file_ext = img_path.split('.')[-1]
    assert file_ext in ['png', 'jpg', 'jpeg'], 'The input must be an image'

    image = cv2.imread(img_path)
    assert image is not None, f'Error while loading the image {img_path}'

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_images(path):
    images_paths = list_files(path)
    images = [load_image(p) for p in images_paths]
    return images
