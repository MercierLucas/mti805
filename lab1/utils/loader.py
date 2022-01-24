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


def list_files_in_dir_recursive(path):
    """List all images in subdirs"""
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


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    # return the resized image
    return resized
