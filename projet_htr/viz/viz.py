import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from math import ceil


def show_predicted(img, label, predicted):
    plt.imshow(img)

    plt.title(f'Label was {label}, model predicted {predicted}')
    plt.show()


def plot_conf_matrix(preds, trues, labels):
    mat = confusion_matrix(trues, preds, labels=labels)
    sns.heatmap(mat, annot=True)
    plt.show()

def plot_acc_over_infer_time(models):
    for name, infos in models.items():
        plt.plot(infos['train_time'], infos['accuracy'], 'x')
        plt.text(infos['train_time'], infos['accuracy'], name)

    plt.xlabel('Train time in seconds')
    plt.ylabel('Accuracy')

    plt.show()



def show_images(images, labels, rows='auto', cols='auto', max_cols=4, size=None, axis=False):
    """Show images side-by-side"""

    assert len(images) == len(labels), 'Must have same number of labels and images'
    
    if size:
        plt.figure(figsize=(size,size))
    
    if rows == 'auto' and cols == 'auto':
        cols = max_cols if len(images) > max_cols else len(images)
        rows = ceil(len(labels) / cols)
        
    for i, (image, label) in enumerate(zip(images, labels)):
        plt.subplot(rows, cols, (i+1))
        if not axis:
            plt.axis('off')
        if len(image.shape) == 2:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)
        plt.title(label)
    plt.show()