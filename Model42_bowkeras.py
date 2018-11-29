import platform as plat
import re  # regex
from tqdm import tqdm
import itertools
import tfplot
import matplotlib
from sklearn.metrics import confusion_matrix
from textwrap import wrap

from general_func.file_wav import *
from general_func.gen_func import Comapare2

# LSTM_CNN
import keras as kr
import numpy as np
import random
import tensorflow as tf

from readdata_bowel import DataSpeech
from readdata_bowel import MAX_AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, CLASS_NUM
from help_func.FL import focal_loss
from collections import Counter

