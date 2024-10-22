import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import os
from sklearn.metrics import confusion_matrix

# import scikitplot as skplt


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

    plt.figure(figsize=(10, 8))
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
    plt.savefig('plots/cm.png')


def plot_conf_matrix(y_test, y_pred, pic_name, norm=True):
    fig = plt.figure(figsize=(10, 8))
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=norm)
    plt.savefig('plots/'+pic_name+'.png')


def plot_metrics(history, image_name):
    mpl.rcParams['figure.figsize'] = (12, 10)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metrics = ['loss', 'f1_score', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n+1)
        plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[1], linestyle="--", label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc_score':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])
        plt.legend()
    plt.suptitle('Results Metrics - ' + image_name)
    plt.savefig('plots/'+image_name + '.png')
    plt.show()


def plot_cm(cm, model_name):
    fig = plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    #plt.title('Confusion Matrix - ', model_name)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig('plots/cm_' + model_name + '.png')
    plt.show()


def plot_loss_accuracy(history, pic_name):
    fig = plt.figure(figsize=(10, 8))
    # plot loss during training
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/'+pic_name+'.png')


def plot_loss_metric(history, model_name):
    fig = plt.figure(figsize=(10, 8))
    # plot loss during training
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(list(history.history.values())[2], label='train')
    plt.plot(list(history.history.values())[0], label='test')
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title(str(list(history.history.keys())[3]))
    plt.plot(list(history.history.values())[3], label='train')
    plt.plot(list(history.history.values())[1], label='test')
    plt.legend()
    pic_name = str(list(history.history.keys())[3])
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/loss_'+pic_name+'_'+model_name+'.png')


def plot_clf_report(clf_report, pic_name):
    # .iloc[:-1, :] to exclude support
    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
    plt.savefig('plots/'+pic_name+'.png')


def plot_pre_rec_thresh(precision, recall, thresholds, pic_name, figsize=(10, 8)):
    """

    :param precision:
    :param recall:
    :param figsize:
    :return:
    """
    fig = plt.figure(figsize=(10, 8))
    plt.plot(thresholds, precision[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recall[:-1], 'g--', label='Recall')
    fig.suptitle("Precision Vs. Recall", fontsize=18)
    plt.xlabel('Threshold')
    plt.legend(loc='upper right')
    plt.ylim([0, 1])
    plt.savefig('plots/'+pic_name+'.png')


def plot_pre_rec(y_test, y_proba, pic_name):
    fig = plt.figure(figsize=(10, 8))
    skplt.metrics.plot_precision_recall(y_test, y_proba)
    plt.savefig('plots/'+pic_name+'.png')


def plot_roc(y_test, y_proba, pic_name):
    fig = plt.figure(figsize=(10, 8))
    skplt.metrics.plot_roc(y_test, y_proba)
    plt.savefig('plots/'+pic_name+'.png')


def plot_roc_curve(roc_auc, fpr, tpr, pic_name):
    fig = plt.figure(figsize=(10, 8))
    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    # set title & xlabel & ylabel
    fig.suptitle("ROC AUC Curve | AUC = " + str(roc_auc), fontsize=18)
    plt.xlabel('FalsePositiveRate', fontsize=15)
    plt.ylabel('TruePositiveRate', fontsize=15)
    # show the plot
    plt.savefig('plots/'+pic_name+'.png')


# =================== Exploratory Data Analysis (EDA) =====================
def plot_time_series(input_df, columns_list, title, pic_name):
    fig = input_df[columns_list].plot(marker='_',
                                      alpha=0.5,
                                      linestyle='None',
                                      figsize=(15, 15),
                                      subplots=True,
                                      title=title)
    plt.savefig('../eda/eda_plots/' + pic_name + '.png')
    plt.show()


def plot_boxplot(input_df, columns_list):
    fig = input_df[columns_list].plot(kind='box',
                 color=dict(boxes='r', whiskers='r', medians='r', caps='r'),
                 boxprops=dict(linestyle='-', linewidth=1.5),
                 flierprops=dict(linestyle='-', linewidth=1.5),
                 medianprops=dict(linestyle='-', linewidth=1.5),
                 whiskerprops=dict(linestyle='-', linewidth=1.5),
                 capprops=dict(linestyle='-', linewidth=1.5),
                 showfliers=False, grid=True, rot=0, figsize=(15, 8))
    plt.figure(figsize=(10, 8))
    return fig

