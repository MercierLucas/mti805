import time
import json
import string
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models, losses
from sklearn import metrics

from classifiers import LeNet, SVM
from datasets import MNISTDataset, EMNISTDataset, CustomDataset, EMNISTLettersDataset, EMNISTBothDataset
from viz.viz import show_predicted, plot_acc_over_infer_time, plot_conf_matrix, show_images
from trainer import Trainer


def save_json(output, obj):
    with open(f'{output}.json', 'w') as f:
        json.dump(obj, f, sort_keys=True, indent=4)
        print(f'Results saved to {output}.json')


def predict(model, x, y):
    predicted = model.predict_classes(np.array([x]))[0]
    show_predicted(x, y, predicted)


def exec_time(func, arg):
    t = time.time()
    func(arg)
    return time.time() - t


def train():
    limit = 1000
    #datasets = [MNISTDataset(limit=limit), EMNISTDataset(limit=400)]
    datasets = [EMNISTLettersDataset(limit=400)]
    models = [SVM('linear'), SVM('rbf')]
    train = Trainer(datasets)
    train.train(models, output='stats')


def evaluate_dataset(dataset, models):
    results = {}
    print(f'Testing on {dataset.name}')
    for m in models:
        #t = exec_time(m.evaluate, dataset)
        print(f'DEBUT {m.name}')
        acc = m.evaluate(dataset)
        #preds = [m.predict(x) for x in dataset.test_x]
        print(f'FIN {m.name}')
        #if max(preds) > 26:
        #    y_true = dataset.test_y_emnist
        #else:
        #    y_true = dataset.test_y
        #acc = metrics.accuracy_score(y_true=y_true, y_pred=preds)
        results[m.name] = acc
        #print(f'> {m.name} took {t}s to infer. Accuracy: {m.accuracy}')
    return results


def benchmark():
    models_infos = {}

    mnist_models = [
        #SVM('linear', model='./models/SVClinear_MNIST.pkl'),
        #SVM('rbf', model = './models/SVCrbf_MNIST.pkl'),
        LeNet('./models/lenet_mnist09779.h5'),
        LeNet('./models/bncnn_mnist_09943.h5', name='bncnn_mnist'),
    ]

    emnist_models = [
        #SVM('linear', model='./models/SVClinear_EMNIST_letters.pkl'),
        #SVM('rbf', model = './models/SVCrbf_EMNIST_letters.pkl'),
        LeNet('./models/lenet_emnist09016.h5', name='lenet_letters'),
        LeNet('./models/bncnn_letters_09448.h5', name='bncnn_letters'),
    ]

    emnist_both_models = [
        LeNet('./models/lenet_byclass_08615.h5', name='lenet_both'),
        LeNet('./models/bncnn_byclass_08711.h5', name='bncnn_both'),

    ]
    
    models_infos['mnist'] = evaluate_dataset(MNISTDataset(limit=None), mnist_models)
    models_infos['emnist_letters'] = evaluate_dataset(EMNISTLettersDataset(limit=None), emnist_models)
    models_infos['custom'] = evaluate_dataset(CustomDataset('./custom_dataset/alphabet'), emnist_models)
    models_infos['both'] =  evaluate_dataset(EMNISTBothDataset(), emnist_both_models)

    save_json('benchmark_nets', models_infos)
    #plot_acc_over_infer_time(models_infos)


def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def detect():
    display = 'split'
    #img = cv2.imread('./custom_dataset/mots_majuscule.jpg')
    lenet = LeNet('models/lenet_emnist09016.h5', classif_type='letters')
    #lenet = LeNet('models/lenet_mnist_09933.h5')
    #img = cv2.imread('custom_dataset/full_letters.jpg')
    img = cv2.imread('custom_dataset/sample1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #labels = list(range(10)) + list (string.ascii_letters)

    grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, grayscale = cv2.threshold(grayscale, 125, 255, cv2.THRESH_BINARY_INV)

    edges = cv2.Canny(grayscale, 30, 200)

    rois = []
    preds = []

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for idx, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if w > 4*h or h > 3*w or w < 20 or h < 20:
            continue
        roi = img[y:y+h, x:x+w]
        roi = preprocess_real(roi, input_mode='RGB')
        pred = lenet.predict(roi, return_class=True)

        if display == 'split':
            rois.append(roi)
            preds.append(pred)
        cv2.rectangle(img, (x,y), (x+w,y+h), (200,0,0), 2)
        cv2.putText(img, f'{pred}', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX,1, (200,0,0), 1, 2)

    if display == 'split':
        show_images(rois, preds, axis=False)

    elif display == 'overlay':
        plt.imshow(img)
        plt.show()


def preprocess_real(img, input_mode='BGR'):
    if input_mode == 'BGR':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif input_mode == 'RGB':
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    _, gray = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)
    gray = cv2.resize(gray, (28, 28))
    gray = np.pad(gray, 2, mode='constant')
    gray = normalize(gray)
    return gray


def preprocess_mnist(img):
    return tf.pad(img, [[0, 0], [2, 2], [2, 2]]) / 255


def compare_preprocessing():
    real = cv2.imread('./custom_dataset/alphabet/A.jpg')
    real = preprocess_real(real)

    ds = EMNISTDataset()
    letter = ds.get_char_sample('a') #get_emnist_letter(10, know_idx=309)
    lenet = LeNet('models/lenet_byclass_07135.h5')
    pred = lenet.predict(real)
    pred = ds.get_label(pred)

    show_images([real, letter], [f'Real A, Pred: {pred}', 'EMNIST'], axis=True)



def display_preprocessing():
    real = cv2.imread('./custom_dataset/alphabet/F.jpg')
    real = preprocess_real(real)
    labels = list(range(10)) + list (string.ascii_letters)
    lenet = LeNet('models/lenet_byclass_07135.h5')
    pred = lenet.predict(real)
    pred = labels[pred]
    show_images([real], [f'Real A, Pred: {pred}'], axis=True)


def show_custom_dataset():
    ds = CustomDataset('./custom_dataset/alphabet')
    classif = LeNet('./models/bncnn_letters_09448.h5', name='bncnn_letters')
    #classif = SVM('linear', model='./models/SVCrbf_EMNIST.pkl')
    preds = []

    for img in ds.test_x:
        pred = classif.predict(img)
        pred_label = ds.get_label(pred)
        preds.append(f'{pred_label}')# ({pred})')

    show_images(ds.test_x, preds, axis=False)


def split_letters():
    img = cv2.imread('./custom_dataset/mots_majuscule.jpg')
    lenet = LeNet('models/lenet_emnist09016.h5')
    grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, grayscale = cv2.threshold(grayscale, 200, 255, cv2.THRESH_BINARY_INV)
    labels = list (string.ascii_uppercase)
    #labels = list(range(28))
    #grayscale = cv2.Canny(grayscale, 30, 200)

    contours, _ = cv2.findContours(grayscale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    preds = []
    for idx, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if w < 5 or h < 5:
            continue
        #if w > 4*h or h > 3*w or w < 20 or h < 20:
        #    continue
        roi = img[y:y+h, x:x+w]
        roi = preprocess_real(roi, input_mode='RGB')
        pred = lenet.predict(roi) - 1
        pred = labels[pred]
        preds.append(pred)
        #if display == 'split':
        #    rois.append(roi)
        #    preds.append(pred)
        cv2.rectangle(img, (x,y), (x+w,y+h), (200,0,0), -1)
        cv2.putText(img, f'{pred}', (x,y + 20), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 5, 5)

    print(preds)
    show_images([img, grayscale], ['img', 'grayscale'], axis=False)


if __name__ == '__main__':
    task = 'benchmark'
    tasks = {
        'benchmark': benchmark,
        'detect': detect,
        'split_letters': split_letters,
        'compare_preprocessing': compare_preprocessing,
        'show_custom_dataset': show_custom_dataset,
        'train': train,
        'display_preprocessing': display_preprocessing
    }

    tasks[task]()


 