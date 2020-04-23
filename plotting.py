import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools


def plot_confusion_matrix(cm, target_names, normalize=True):
    """
    function to plot the confusion matrix
    :param cm: confusion matrix
    :param target_names: labels names
    :param normalize: whether to normalize or not
    :return: the confusion matrix chart
    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix', fontdict={'family': 'arial', 'weight': 'bold', 'size': 14})
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float')
        # cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100

    font = {'family': 'arial',
            'color': 'black',
            'weight': 'normal',
            'size': 15,
            }

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.0f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black", fontdict=font)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black", fontdict=font)

    plt.tight_layout()
    plt.ylabel('True Labels', fontdict=font)
    plt.xlabel('Predicted Labels\n\nAccuracy={:0.2f}%'.format(accuracy, misclass), fontdict=font)
    #plt.savefig('test_plots/cm_leipzig.png')
    return fig



#df = pd.read_csv('clean_data/P812_M050_B_FoG_trial_1_cleaned.csv', sep=',')


def plot_time_series(input_df, columns_list, title, fig_size=(15, 15)):
    fig = input_df[columns_list].plot(marker='.', alpha=0.5, linestyle='None', figsize=fig_size, subplots=True, title=title)
    return fig


def plot_boxplot(input_df, columns_list, figsize=(15, 8), showfliers=False):
    fig = input_df[columns_list].plot(kind='box',
                 color=dict(boxes='r', whiskers='r', medians='r', caps='r'),
                 boxprops=dict(linestyle='-', linewidth=1.5),
                 flierprops=dict(linestyle='-', linewidth=1.5),
                 medianprops=dict(linestyle='-', linewidth=1.5),
                 whiskerprops=dict(linestyle='-', linewidth=1.5),
                 capprops=dict(linestyle='-', linewidth=1.5),
                 showfliers=False, grid=True, rot=0, figsize=(15, 8))
    return fig

