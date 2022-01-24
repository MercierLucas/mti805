# Lab 1: face id


## Usage

### Train
```bash
python main.py \\
    '--action' 'train' \\
    '--input' 'dataset' \\
    '--pickle_dir' 'pickle' \\
    '--weights' 'weights'
```


### Inference (only detection)
```bash
python main.py \\
    '--action' 'face_detection' \\
    '--input' 'webcam' \\
    '--pickle_dir' 'pickle' \\
    '--weights' 'weights'
```

### Inference (detection and identification)
In this case input can be *webcam* or a path to an image
```bash
python main.py \\
    '--action' 'detect_and_id' \\
    '--input' 'webcam' \\
    '--pickle_dir' 'pickle' \\
    '--weights' 'weights'
```

## Weights
Weights can be found here:
- [XML files for CascadeClassifier](https://github.com/opencv/opencv/tree/master/data/haarcascades)
- [Torch weights trained on OpenFace](https://github.com/pyannote/pyannote-data)