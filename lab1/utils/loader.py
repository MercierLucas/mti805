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
