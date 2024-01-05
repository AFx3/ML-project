import numpy as np
from numpy.random import standard_normal, normal, uniform
import time
from keras import backend as K


def _mean_squared_error(y, pred):
    return 0.5 * np.mean((y - pred) ** 2)


def _mean_abs_error(y, pred):
    return np.mean(np.abs(y, pred))


def _euclidean_error(y_true, y_pred):
    return np.sum(np.sqrt(np.sum(np.square(y_true - y_pred), axis=1))) / len(y_true)


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def _tanh(x):
    return np.tanh(x)


def _fourier(x):
    return np.sin(x)


def _hardlimit(x):
    return (x >= 0).astype(int)


def _relu(x):
    return np.maximum(0, x)


def _identity(x):
    return x


def _standard_normal(size):
    return standard_normal(size)


def _xavier(size):
    bound = (np.sqrt(6) / np.sqrt(size[0] + size[1]))
    return uniform(low=-bound, high=bound, size=size)


def _he_normal(size):
    return normal(loc=0.0, scale=np.sqrt(2 / size[0]), size=size)


def getActivation(name):
    return {
        'sigmoid': _sigmoid,
        'fourier': _fourier,
        'hardlimit': _hardlimit,
        'tanh': _tanh,
        'relu': _relu
    }[name]


def getLoss(name):
    return {
        'mse': _mean_squared_error,
        'mae': _mean_abs_error,
        'mee': _euclidean_error
    }[name]


def getInit(name: str, size: tuple):
    return {
        'std': _standard_normal(size),
        'xavier': _xavier(size),
        'he': _he_normal(size)
    }[name]


class ELM:
    def __init__(self, num_input_nodes, num_hidden_units, num_out_units, activation='sigmoid',
                 loss='mse', beta_init=None, w_init=None, bias_init=None):
        self._num_input_nodes = num_input_nodes
        self._num_hidden_units = num_hidden_units
        self._num_out_units = num_out_units

        self._activation = getActivation(activation)
        self._loss = getLoss(loss)

        if isinstance(beta_init, np.ndarray):
            self._beta = beta_init
        else:
            self._beta = np.random.uniform(-1., 1., size=(self._num_hidden_units, self._num_out_units))

        if isinstance(w_init, str):
            self._w = getInit(w_init, size=(self._num_input_nodes, self._num_hidden_units))
        else:
            self._w = np.random.uniform(-1, 1, size=(self._num_input_nodes, self._num_hidden_units))

        if isinstance(bias_init, np.ndarray):
            self._bias = bias_init
        else:
            self._bias = np.zeros(shape=(self._num_hidden_units,))

        print('Bias shape:', self._bias.shape)
        print('W shape:', self._w.shape)
        print('Beta shape:', self._beta.shape)

    def fit(self, X, Y, display_time=False):
        H = self._activation(X.dot(self._w) + self._bias)

        # Moore–Penrose pseudo inverse
        if display_time:
            start = time.time()
        H_pinv = np.linalg.pinv(H)
        if display_time:
            stop = time.time()
            print(f'Train time: {stop-start}')

        self._beta = H_pinv.dot(Y)

    def __call__(self, X):
        H = self._activation(X.dot(self._w) + self._bias)
        return H.dot(self._beta)

    def evaluate(self, X, Y):
        pred = self(X)

        # Loss (based on model setting)
        loss = self._loss(Y, pred)

        # Accuracy
        acc = np.sum(np.argmax(pred, axis=-1) == np.argmax(Y, axis=-1)) / len(Y)

        return loss, acc
