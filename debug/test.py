from textwrap import wrap
import re
import itertools
import tfplot
import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import os
# -*- coding: utf-8 -*-


def plot_confusion_matrix(correct_labels, predict_labels, labels, title='Confusion matrix', tensor_name = 'MyFigure/image', normalize=False):
    '''
    Parameters:
        correct_labels                  : These are your true classification categories.
        predict_labels                  : These are you predicted classification categories
        labels                          : This is a lit of labels which will be used to display the axix labels
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor

    Returns:
        summary: TensorFlow summary

    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc.
        - Currently, some of the ticks dont line up due to rotations.
    '''
    cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
    if normalize:
        cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predictions ', fontsize=25)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=20, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('Truths', fontsize=25)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=20, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=25, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
    summary = tfplot.figure.to_summary(fig, tag=tensor_name)  #Convert a matplotlib figure fig into a TensorFlow Summary object that can be directly fed into Summary.FileWriter
    return summary


map = {0:'normal',1:'wheeze',2:'crackle',3:'both'}
checkpoint_dir = '../checkpoints'
y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
true = ['normal', 'crackle', 'crackle', 'wheeze', 'normal', 'wheeze', 'normal', 'crackle', 'crackle', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'wheeze', 'normal', 'wheeze', 'normal', 'wheeze', 'crackle', 'crackle', 'crackle', 'crackle', 'crackle', 'crackle', 'normal', 'crackle', 'normal', 'normal', 'crackle', 'normal', 'both', 'normal', 'wheeze', 'normal', 'normal', 'normal', 'normal', 'crackle', 'normal', 'normal', 'crackle', 'wheeze', 'crackle', 'normal', 'crackle', 'both', 'normal', 'normal']
predict = ['wheeze', 'both', 'both', 'both', 'normal', 'both', 'both', 'both', 'both', 'both', 'normal', 'normal', 'both', 'normal', 'normal', 'normal', 'both', 'both', 'both', 'normal', 'both', 'both', 'both', 'both', 'normal', 'both', 'both', 'both', 'normal', 'both', 'both', 'both', 'wheeze', 'both', 'both', 'normal', 'both', 'normal', 'both', 'both', 'normal', 'both', 'both', 'both', 'both', 'both', 'normal', 'both', 'both', 'both']
llaabb = list(map.values())
y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
labels=["ant", "bird", "cat"]
os.system('rm -rf ../checkpoints/summaries ')
with tf.Session() as  sess:
    img_d_summary_dir = os.path.join(checkpoint_dir, "summaries", "img")
    img_d_summary_writer = tf.summary.FileWriter(img_d_summary_dir, sess.graph)
    # img_d_summary = plot_confusion_matrix(y_true, y_pred, labels, tensor_name='dev/cm')
    img_d_summary = plot_confusion_matrix(true, predict, llaabb, tensor_name='dev/cm')
    img_d_summary_writer.add_summary(img_d_summary, global_step=50)

