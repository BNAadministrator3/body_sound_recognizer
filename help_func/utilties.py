from keras.layers.recurrent import GRU, LSTM
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.layers.core import Activation, Dense, Dropout

def get_activation(act, string=False):
    str_act = ['relu', 'tanh', 'sigmoid', 'linear','softmax','softplus','softsign','hard_sigmoid']
    if (act in str_act):
        if string:
            return act
        else:
            return Activation(act)
    else:
        return {'prelu': PReLU(), 'elu' : ELU(), 'lrelu' : LeakyReLU(),
               }[act]

def get_rnn(name, units, act='tanh', retseq=True, inp=None):
    if (name == 'gru'):
        if (inp is not None):                                       #only return strings
            return GRU(units, return_sequences = retseq, activation=get_activation(act, True), input_shape=inp)
        else:
            return GRU(units, return_sequences = retseq, activation=get_activation(act, True)) #adopt this
    else:
        if (inp is not None):
            return LSTM(units, return_sequences = retseq, activation=get_activation(act, True), input_shape=inp)
        else:
            return LSTM(units, return_sequences = retseq, activation=get_activation(act, True))