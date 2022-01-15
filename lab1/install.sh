#!/usr/bin/env bash


#caffee model
mkdir weights
wget -nc -O weights/res10_300x300_ssd_iter_140000.caffemodel https://github.com/gopinath-balu/computer_vision/blob/master/CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel?raw=true
wget -nc -O weights/deploy.prototxt https://raw.githubusercontent.com/gopinath-balu/computer_vision/master/CAFFE_DNN/deploy.prototxt.txt

#torch openface
wget -nc -O weights/openface.nn4.small2.v1.t7 https://github.com/pyannote/pyannote-data/blob/master/openface.nn4.small2.v1.t7?raw=true


