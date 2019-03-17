# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from six.moves import xrange
import sys
import numpy as np

def cost(prob):
    """Return the negative log-likelihood."""
    return -np.mean(np.log(prob))


def accuracy(y, y_):
    """Return the accuracy."""
    return np.sqrt(np.mean((y - y_)**2))


#from scikit-learn neural network
def log_loss(y_true, y_prob):
    """Compute Logistic loss for classification.

    Parameters:
        y_true : array-like or label indicator matrix as ground truth (correct) labels.

        y_prob : array-like of float, shape = (n_samples, n_classes)
            Predicted probabilities, as returned by a classifier's
            predict_proba method.

    Returns:
        loss : float, The degree to which the samples are correctly predicted.
    """
    y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)

    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)

    if y_true.shape[1] == 1:
        y_true = np.append(1 - y_true, y_true, axis=1)

    return -np.sum(y_true * np.log(y_prob)) / y_prob.shape[0]


def binary_log_loss(y_true, y_prob):
    """Compute binary logistic loss for classification.

    This is identical to log_loss in binary classification case,
    but is kept for its use in multilabel case.

    Parameters:
        y_true : array-like or label indicator matrix.
            Ground truth (correct) labels.

        y_prob : array-like of float, shape = (n_samples, n_classes)
            Predicted probabilities, as returned by a classifier's
            predict_prob method.

    Returns:
        loss : float
            The degree to which the samples are correctly predicted.
    """
    y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)

    return -np.sum(y_true * np.log(y_prob) +
                   (1 - y_true) * np.log(1 - y_prob)) / y_prob.shape[0]


def xavier_initializer(n_in, n_out, uniform=True, dtype=None):
    """Initializer performing "Xavier" initialization for weights matrix.

    This initializer is designed to keep the scale of the gradients roughly
    the same in all layers. In uniform distribution this ends up being the
    range: `x = sqrt(6. / (in + out)); [-x, x]` and for normal distribution
    a standard deviation of `sqrt(3. / (in + out))` is used.

    Arguments:
        uniform: Whether to use uniform or normal distributed random
            initialization.
        seed: A Python integer. Used to create random seeds. See
            `set_random_seed` for behavior.
        dtype: The data type. Only floating point types are supported.

    Return:
        An initializer for a weight matrix.

    Reference:
        Xavier Glorot and Yoshua Bengio. Understanding the difficulty of
        training deep feedforward neural networks.  In Proceedings of the
        13th International Conference on Artificial Intelligence and
        Statistics (AISTATS), vol. 9, Chia Laguna Resort, Sardinia, Italy,
        2010.

    Links:
        [http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf]
        (http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    """
    high = np.sqrt(6. / (n_in + n_out))
    low  = - high
    if dtype == None:
        dtype = np.float32
    return np.array(np.random.uniform(low,high,size=(n_in,n_out)),dtype=dtype)


def xavier_initializer_wb(n_in, n_out, uniform=True, dtype=None):
    """Xaivier initialzer both weights and biases."""

    high = np.sqrt(6. / (n_in + n_out))
    low  = - high
    if dtype == None:
        dtype = np.float32
    tmp = np.array(np.random.uniform(low,high,size=(n_in+1,n_out)),dtype=dtype)
    w = tmp[:-1,:]
    bias = tmp[-1:,:]
    return w, bias


class relu(object):
    """Rectified linear Unit.

    Reference:
        Vinod Nair and Geoffrey E. Hinton. Rectified linear units
        improve restricted boltzmann machines. In Johannes F\"{u}rnkranz
        and Thorsten Joachims, (eds.), In Proceedings the 27th International
        Conference on Machine Learning (ICML), Haifa, Israel, June 2010. Omnipress.
    """

    @staticmethod
    def forward(x):
        #return 0 if x.any() < 0 else x.any()
        #return max(0, x)
        #return np.max(0, x, axis=None, keepdims=True)
        #return np.max(0, x)
        #return x * (x > 0)
        #return np.clip(x,0,np.finfo(x.dtype).max),out=x)
        return np.clip(x,0,np.finfo(x.dtype).max)

    @staticmethod
    def backward(x):
        return (x > 0) * 1.
        # return (x[0] > 0) * 1.


class sigmoid(object):
    """Logistic Sigmoid function, that returns from 0 to 1.

    Reference:
        David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. williams.
        Learning representations by back-propagating errors. Nature,
        323(6088):533-536, 1986

    Link:
        [https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/].
    """

    @staticmethod
    def forward(x):
        if x.any() >= 0:
            z = np.exp(x)
        else:
            z = np.exp(-x)
        return z / (1 + z)
        #return 1 / (1. + np.exp(-x))

    @staticmethod
    def backward(y):
        #y = sigmoid.forward(x)
        #return x * (1. - x)
        return y * (1. - y)


class tanh(object):
    """Hyper tangent.

    Reference:
        LeCun, Y., Bottou, L., Orr, G. B., & Muller, K.-R. (1998).
        Efficient backprops. In G. Montavon, G. B. Orr, K.-R. Muller (Eds.),
        Neural networks: tricks and the trade (p. 9-48). Berlin Heidelberg,
        Germany: Springer-Verlag.
    """
    @staticmethod
    def forward(x):
        return np.tanh(x)
        # return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    @staticmethod
    def backward(y):
        #y = np.tanh(x)
        return 1. - y**2
        #return 1. - x**2


class softsign:
    @staticmethod
    def forward(x):
        return x / (1. + np.abs(x))

    @staticmethod
    def backward(x):
        return np.sign(x)


class identity:
    @staticmethod
    def forward(x):
        return x

    @staticmethod
    def backward(x):
        return 1.


class sat(object):
    @staticmethod
    def forward(x):
        return np.clip(x, -1,1)

    @staticmethod
    def backward(x):
        return ((x < 1) * 1.) * ((x > -1) * 1.)


class clu(object):
    """Continious Logic Unit.

    References:
        Werbos (1990) Backpropagation Through Time: What It Does and How to Do it.
        In Proceedings of the IEEE, Vol. 78(10), 1550-1560
    """
    @staticmethod
    def forward(x):
        return np.clip(x, 0, 1)

    @staticmethod
    def backward(x):
        return ((x < 1) * 1.) * ((x > 0) * 1.)

 
class softplus(object):
    @staticmethod
    def forward(x):
        return np.log(1+np.exp(x))

    @staticmethod
    def backward(x):
        return sigmoid.forward(x)


class arctan(object):
    @staticmethod
    def forward(x):
        return np.arctan(x)

    @staticmethod
    def backward(x):
        return 1/(1+x**2)


class elu(object):
    """Exponential linear unit.

    References:
        arXiv:1511.07289v5 [cs.LG]
    """
    @staticmethod
    def forward(x):
        y = x.copy()
        neg_indecies = x < 0
        y[neg_indecies] = (np.exp(y[neg_indecies]) - 1)
        return y

    @staticmethod
    def backward(x):
        tmp = [np.exp(a) if a <= 0 else 1.0 for a in x[0]]
        return tmp


class softmax(object):
    """Softmax function.

    Backpropagation as a maximum likelihood procedure

    If we interpret each output vector as a specification of a conditional
    probability distribution over a set of output vectors given an input
    vector, we can interpret the backpropagation learning procedure as a
    method of finding weights that maximize the likelihood of generating
    the desired conditional probability distributions. Two examples of this
    kind of interpretation will be described.

    Suppose we only attach meaning to binary output vectors and we treat a
    real-valued output vector as a way of specifying a probability
    distribution over binary vectors. We imagine that a real-valued output
    vector is stochastically converted into a binary vector by treating the
    real values as the probabilities that individual components have value
    $1$, and assuming independence between components. For simplicity, we
    can assume that the desired vectors used during training are binary
    vectors, though this is not necessary. Given a set of training cases,
    it can be shown that the likelihood of producing exactly the desired
    vectors is maximized when we minimize the cross-entropy,$C$ , between
    the desired and actual conditional probability distributions:

    $$
    C=-\sum_{j,c}d_{j,c}\log(y_{j,c})+\left(1-d_{j,c}\right)\log\left(1-y_{j,c}\right),
    $$

    where $d_{j,c}$ is the desired probability of output unit $j$ in case 
    $c$ and $y_{j,c}$ is its actual probability.

    Reference:
        Hinton, G. E. (1989). Connectionist learning procedures.
        Artificial Intelligence, 40, 185{234.

    """
    @staticmethod
    def forward(X):
        out = np.exp(X - logsumexp(X, axis=1, keepdims=True))
        return out

    @staticmethod
    def backward(X):
        return 1


class Layer(object):
    """prototype class of any layers."""
    def __init__(self, **kwargs):
        dtype = kwargs.get('dtype')
        if dtype is None:
            dtype = np.float32
        self.dtype = dtype
        name = kwargs.get('name')
        if name is None:
            name = 'Prototype Layer'


class fc_layer(Layer):
    def __init__(self, n_in, n_out, activation=relu, **kwargs):
        dtype = kwargs.get('dtype')
        if dtype is None:
            dtype = np.float32
        self.dtype = dtype

        name = kwargs.get('name')
        if name is None:
            name = 'fc_layer'
        self.name = name
        super(fc_layer, self).__init__(name=name)

        self.W = xavier_initializer(n_in, n_out)
        self.bias = xavier_initializer(1, n_out)
        self.activation = activation


    def forward(self, X):
        affine = np.dot(X, self.W) + self.bias
        return self.activation.forward(affine)


    def backward(self, delta, X):
        tmp = delta * self.activation.backward(delta)
        grad = np.dot(X.T, tmp)
        return grad


from scipy.misc import logsumexp

class softmax_layer(Layer):
    def __init__(self, n_in, n_out, **kwargs):
        name = kwargs.get('name')
        if name is None:
            name = 'softmax_layer'
        self.name = name
        super(softmax_layer, self).__init__(name=name)
        self.W = xavier_initializer(n_in, n_out)
        self.bias = xavier_initializer(1, n_out)
        ##Memory variables for Adagrad
        self.mW = np.zeros_like(self.W)

    def forward(self, inp):
        #See compute_prob function below for detail description
        state = np.dot(inp, self.W) + self.bias
        return self.compute_prob(state)

    #def backward(self, delta, out, inp):
    def backward(self, delta, inp):
        #back_delta = (out * (k - out)).sum(axis=0)
        #back_delta = out
        grad = np.dot(inp.T, delta)
        return grad
        #return err

    @staticmethod
    def compute_prob(x):
        """softmax based probabilities.

        Do not use softmax: np.exp(x)/np.sum(np.exp(x))
        Use `scipy.misc.logsumexp()` instead, in order to compute
        probabilities of each element in `$\mathbf{y}$`
        """
        ret = np.exp(x - logsumexp(x, axis=1, keepdims=True))
        #When you compute log probability of `$\mathbf{y}$`, 
        #You can use `x - scipy.misc.logsumexp(x)`
        return ret

    @staticmethod
    def compute_negative_log_prob(x):
        return -x + logsumexp(x)

    @staticmethod
    def compute_mse(y, y_):
        errsq = (y - y_)**2
        return np.mean(errsq)

    @staticmethod
    def compute_cost(y, y_):
        sumCost = - y * np.log(y_)
        return np.mean(sumCost)
        #return y_ - y


#for fetching mninst data via sklearn
from sklearn.datasets import fetch_mldata
from sklearn import preprocessing
import matplotlib.pyplot as plt

def fetch_mnist(image_num=100):
    mnist = fetch_mldata('MNIST original')
    X = np.float32(mnist.data[:]) / 255.

    if (image_num == 0) or (image_num >= X.shape[0]):
        image_num = X.shape[0]


    idx = np.random.choice(X.shape[0], image_num)
    X = X[idx]
    y = np.int32(mnist.target[idx]).reshape(image_num, 1)
    y = preprocessing.OneHotEncoder(sparse=False).fit_transform(y)

    #train_idx, test_idx = train_test_split(np.array(range(n)), test_size=0.05)
    #train_X, test_X = X[train_idx], X[test_idx]
    #train_y, test_y = y[train_idx], y[test_idx]

    #train_y = preprocessing.OneHotEncoder(sparse=False).fit_transform(train_y)
    #train_y = np.float32(train_y_)
    #X, y = train_X, train_y

    return X, y


def view_mnist(X, y, image_num=0):
    assert image_num <= len(X), \
    '%d you required is larger than image size:%d' % (image_num, len(X))
    a = X[image_num].reshape(28,28) * 255
    label = y[image_num]
    print('label= ', label)
    plt.imshow(a, cmap=plt.cm.gray_r)
    return np.argmax(y[image_num])

def one_hot_vector(i, max):
    ret = np.zeros((max),dtype=np.float32)
    ret[i] = 1
    return ret

def flatten(args):
    #from https://stackoverflow.com/questions/12472338/flattening-a-list-recursively
    #define base case to exit recursive method
    if len(args) == 0:
        return []
    elif isinstance(args, list) and type(args[0]) in [int,str]:
        return [args[0]] + flatten(args[1:])
    elif isinstance(args, list) and isinstance(args[0], list):
        return args[0] + flatten(args[1:])
    else:
        return flatten(args[1:])


class srn_layer(Layer):
    """simple recurrent layer which defines init, forward, and backward functions.

    Simple Recurrent Neural Networks, such as the Jordan and Elman networks.

    References:
        Elman, J. L. (1990). Finding structure in time. Cognitive Science,
            14, 179-211.
        Elman, J. L. (1991). Distributed representations, simple recurrent
            networks, and grammatical structure. Ma chine Learning, 7, 195-225.
        Jordan, M. I. (1986). Serial order: A parallel distributed processing 
            approach (Tech. Rep.). San Diego, CA: University of California, 
            San Diego.
    """
    def __init__(self, n_in, n_hid=None, activation=relu, lr=1e-1, **kwargs):
        name = kwargs.get('name')
        if name is None:
            name = 'srn_layer'
        super(srn_layer, self).__init__(name=name)
        if n_hid is None:
            n_hid = n_in.shape[1]
        self.W = xavier_initializer(n_in, n_hid)
        self.Wr = xavier_initializer(n_hid, n_hid)
        self.bias = xavier_initializer(1, n_hid)
        self.activation = activation
        self.n_hid = n_hid
        self.n_in = n_in

    def forward_a_step(self, x, state):
        affine = np.dot(x, self.W) + np.dot(state, self.Wr) + self.bias
        out = self.activation.forward(affine)
        return out, affine

    def forward(self, X, state=None, hInit=None):
        """Recieve input X, and return values.

        Args:
            X: the input sequnces, X.shape[0] means the input sequnces,
            while X.shape[1] means the dimension of features
            y: the teacher signals
            y_: the output signals
            state: the internal state
        """
        if state is None:
            state = np.zeros(((X.shape[0], self.n_hid)),dtype=np.float32)
        #assert X.shape[0] == state[0], 'X.shape[0] is not equal to state[0]'
        #assert self.n_hid == state[1], 'state[1] is not equal to self.n_hid'
        #assert hInit.shape[1] == self.n_hid, 'hIinit.shape[1] is not equal to self.n_hid'
        #if hInit.all() == None:
            #hInit = np.zeros((1, self.n_hid),dtype=np.float32)
        hInit = np.zeros((1, self.n_hid),dtype=np.float32)
        for t, x in enumerate(X):
            c = np.copy(state[t-1]) if t == 0 else np.copy(hInit)
            state[t], _ = self.forward_a_step(x, c)
            out = self.activation.forward(state)
        return out, state

    def backward(self, X, state, delta):
        """Recieve delta, input X, and the inner `state', then return grad."""
        # Which the word `delta' or `error' should be used?
        # When we woudl consider the original back-propagation algorithm that
        # David Rumelhart et al. proposed 1986, we would like better employ `error'
        # However, they call it `the generalized delta rule' in general, they would like
        # to employ `delta'
        grad = self.activation.backward(delta) * delta
        gradW1 = np.dot(X.T, grad)
        gradWr = np.dot(state.T, grad)
        return gradW1, gradWr, delta


def txt2data(filename='j_constitution9.txt'):
    """Convert the text file to the data.

    Arguments:
        filename: string, default='j_constitution9.txt'.
        Filename will be given by the argument, and converting it to the data matrix X.

    Returns:
        y: target matrix,
        idx2wrd:  dictionary from index to word
        wrd2idx:  dictionary from word to index
    """
    import codecs
    def make_data(vocab):
        wrd2idx, idx2wrd = dict(), dict()
        for i, x in enumerate(vocab):
            wrd2idx[x]=i
            idx2wrd[i]=x
        return wrd2idx, idx2wrd
    
    if not filename:
        #filename = 'okuno_hosomichi.txt'
        #filename = 'ElmanTrain.data'
        filename = 'j_constitution9.txt'
        #filename = 'j_constitutionPreface.txt' 
    contents = []
    print('filename={0}'.format(filename))
    with codecs.open(filename,'r','utf-8') as f:
        data = f.readlines()
        for line in data:
            for word in line.strip():
                for char in word:
                    contents.append(char)
            contents.append('</s>')

    vocab = tuple(sorted(set(contents)))
    wrd2idx, idx2wrd = make_data(vocab)
    vocab_size = len(vocab)
    Xshape0 = len(contents)
    X = np.ndarray((Xshape0, vocab_size),dtype=np.float32)
    y = np.zeros_like(X)
    for t, c in enumerate(contents):
        X[t] = np.copy(one_hot_vector(wrd2idx[c], vocab_size))
    y[-1] = X[0]
    for t, x in enumerate(X[:-1]):
        y[t] = X[t+1]
    return X, y, idx2wrd, wrd2idx

##########################################################################
"""These four functions below would be called from pmsp96."""

def string2onehot_matrix(string, maxLen, wrd2idx):
    """Converts an ASCII string to a one-of-k encoding."""
    arr = np.array([wrd2idx[c] for c in string]).T
    #arr = np.array([one_hot_vector(wrd2idx[ch],maxLen) for ch in string]).T
    ret = np.array(arr[:,None] == np.arange(maxLen)[None, :], dtype=int)
    return ret


def onehot_matrix2string(one_hot_matrix, __idx2wrd):
    """Assume each matrix made of one_hot encoding column vectors."""
    return "".join(__idx2wrd[np.argmax(c)] for c in one_hot_matrix)
    #return "".join([chr(np.argmax(c)) for c in one_hot_matrix])


def make_dict(vocab):
    """Making dictionaries from/to each word to/from index."""
    wrd2idx, idx2wrd = dict(), dict()
    for i, x in enumerate(vocab):
        wrd2idx[x]=i
        idx2wrd[i]=x
    return wrd2idx, idx2wrd


def build_dataset(filename):
    """Loads a text file, and turns each line into an encoded sequence."""
    import codecs

    with codecs.open(filename,'r','utf-8') as f:
        data = f.readlines()
    data = [line.strip() for line in data if len(data) > 2]   # Remove blank lines
    content = []
    #data = data[:max_lines]
    maxLineLength = 0
    for line in data:
        if maxLineLength < len(line):
            maxLineLength = len(line)
        for word in line:
            content.append(word)
        content.append('</s>')
    content.append(' ')
    content.append('　')
    vocab = tuple(sorted(set(content)))
    vocab_size=len(vocab)
    wrd2idx, idx2wrd = make_dict(vocab)
    sequence_length = len(data)
    seqs = np.zeros((sequence_length, maxLineLength, vocab_size))
    #seqs has the elements of 1line's length, number of total lines, vocab size
    for i, line in enumerate(data):
        padded_line = (line + " " * maxLineLength)[:maxLineLength]
        #padded_line = (' ' * (maxLineLength - len(line)) + line)[:maxLineLength]
        seqs[i,:,:] = string2onehot_matrix(padded_line, vocab_size, wrd2idx)
    return seqs, data, wrd2idx, idx2wrd


