# Lab 1: face id


## Usage

### Train
```bash
python main.py \\
    '--faceid_action' 'train' \\
    '--dataset' 'dataset' \\
    '--pickle_dir' 'pickle' \\
    '--weights' 'weights'
```


### Inference (only faceid)
```bash
python main.py \\
    '--faceid_action' 'recognize' \\
    '--input' 'test/sample.jpg' \\
    '--dataset' 'dataset' \\
    '--pickle_dir' 'pickle' \\
    '--weights' 'weights'
```

### Inference
In this case input can be *webcam* or a path to an image
```bash
python main.py \\
    '--faceid_action' 'recognize' \\
    '--add_detector' \\
    '--input' 'test/sample.jpg' \\
    '--dataset' 'dataset' \\
    '--pickle_dir' 'pickle' \\
    '--weights' 'weights'
```