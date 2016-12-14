from __future__ import absolute_import

from .._compat import basestring
from .._compat import chain_exception
from .._compat import pickle
from collections import OrderedDict
import itertools
from warnings import warn
from time import time

import lasagne
from lasagne.layers import get_all_layers
from lasagne.layers import get_output
from lasagne.layers import InputLayer
from lasagne.layers import Layer
from lasagne.layers import InverseLayer
from lasagne.layers import BatchNormLayer, batch_norm
from lasagne.layers.dnn import Conv2DDNNLayer

from lasagne import regularization
from lasagne.objectives import aggregate
from lasagne.objectives import categorical_crossentropy
from lasagne.objectives import squared_error
from lasagne.updates import nesterov_momentum
from lasagne.updates import adam
from lasagne.utils import floatX
from lasagne.utils import unique
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import theano
from theano import tensor as T

import matplotlib.pyplot as plt
from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib as mpl
from collections import namedtuple
import os
import h5py

from . import PrintLog
from . import PrintLayerInfo


class _list(list):
    pass


class _dict(dict):
    def __contains__(self, key):
        return True


def _sldict(arr, sl):
    if isinstance(arr, dict):
        return {k: v[sl] for k, v in arr.items()}
    else:
        return arr[sl]


class Layers(OrderedDict):
    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values()).__getitem__(key)
        elif isinstance(key, slice):
            items = list(self.items()).__getitem__(key)
            return Layers(items)
        else:
            return super(Layers, self).__getitem__(key)

    def keys(self):
        return list(super(Layers, self).keys())

    def values(self):
        return list(super(Layers, self).values())


class BatchIterator(object):
    def __init__(self, batch_size, shuffle=False, seed=42):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random = np.random.RandomState(seed)

    def __call__(self, X, y=None):
        if self.shuffle:
            X0 = X
            if isinstance(X, dict):
                X0 = list(X.values())[0]
            indices = self.random.permutation(np.arange(X0.shape[0]))
            X = _sldict(X, indices)
            if y is not None:
                y = y[indices]
        self.X, self.y = X, y
        return self

    def __iter__(self):
        bs = self.batch_size
        for i in range((self.n_samples + bs - 1) // bs):
            sl = slice(i * bs, (i + 1) * bs)
            Xb = _sldict(self.X, sl)
            if self.y is not None:
                yb = self.y[sl]
            else:
                yb = None
            yield self.transform(Xb, yb)

    @property
    def n_samples(self):
        X = self.X
        if isinstance(X, dict):
            return len(list(X.values())[0])
        else:
            return len(X)

    def transform(self, Xb, yb):
        return Xb, yb

    def __getstate__(self):
        state = dict(self.__dict__)
        for attr in ('X', 'y',):
            if attr in state:
                del state[attr]
        return state


def grad_scale(layer, scale):
    for param in layer.get_params(trainable=True):
        param.tag.grad_scale = floatX(scale)
    return layer


class TrainSplit(object):
    def __init__(self, eval_size, stratify=True):
        self.eval_size = eval_size
        self.stratify = stratify

    def __call__(self, X, y, net):
        if self.eval_size:
            if net.regression or not self.stratify:
                kf = KFold(y.shape[0], round(1. / self.eval_size))
            else:
                kf = StratifiedKFold(y, round(1. / self.eval_size))

            train_indices, valid_indices = next(iter(kf))
            X_train, y_train = _sldict(X, train_indices), y[train_indices]
            X_valid, y_valid = _sldict(X, valid_indices), y[valid_indices]
        else:
            X_train, y_train = X, y
            X_valid, y_valid = _sldict(X, slice(len(y), None)), y[len(y):]

        return X_train, X_valid, y_train, y_valid


class LegacyTrainTestSplit(object):  # BBB
    def __init__(self, eval_size=0.2):
        self.eval_size = eval_size

    def __call__(self, X, y, net):
        return net.train_test_split(X, y, self.eval_size)


def objective(layers,
              loss_function,
              target,
              prediction_transform,
              aggregate=aggregate,
              weights = None,
              deterministic=False,
              l1=0,
              l2=0,
              get_output_kw=None ):
    if get_output_kw is None:
        get_output_kw = {}
    output_layer = layers[-1]

    ###################
    #test_input = np.random.rand(8,1,512,512).astype(np.float32)
    #num_pix = test_input.shape[0]*test_input.shape[2]*test_input.shape[3]
    #layers[0].input_var.tag.test_value = test_input
    ###############
    #print ("Input shape = {}".format(layers[0].input_var.tag.test_value.shape))

    network_output = get_output(
        output_layer, deterministic=deterministic, **get_output_kw)

    #print ("Network output shape={}".format(network_output.tag.test_value.shape))

    ###################
    #test_y = np.random.randint (0,3,(test_input.shape[0],test_input.shape[2],test_input.shape[3]))
    #test_y = np.zeros((test_input.shape[0],test_input.shape[2],test_input.shape[3]))
    #test_y = test_y.astype(np.int32)
    #target.tag.test_value = test_y
    ###############

    #flattened_target =target.flatten()
    #print ("Target shape={}".format(target.tag.test_value.shape))

    '''
    print ("OUTPUTS")
    img = 0
    x = 2
    y = 2
    pixel = y*4 + x
    imgpixel = pixel + img*16
    network_output.tag.test_value[img, 1, x,y] = 10.0
    target.tag.test_value[img,x,y] = 2
    network_output.tag.test_value[img, 1, x+1,y] = 10.0
    target.tag.test_value[img,x+1,y] = 2
    print ("target ={}".format(target.tag.test_value[img,x,y]))
    print ("flat target={}".format(flattened_target.tag.test_value[imgpixel]))
    print ("Orig={}".format(network_output.tag.test_value[img, 0:38, x,y]))

    reshaped_predictions = network_output.flatten(3)
    print ("reshaped_predictions shape={}".format(reshaped_predictions.tag.test_value.shape))
    print ("Flattened={}".format(reshaped_predictions.tag.test_value[img, 0:38, pixel]))

    shuffled = reshaped_predictions.dimshuffle((1,0,2))
    print ("shuffled shape={}".format(shuffled.tag.test_value.shape))
    print ("shuffled={}".format(shuffled.tag.test_value[0:38,img,pixel]))

    flattened = shuffled.flatten(2)
    print ("flattened shape={}".format(flattened.tag.test_value.shape))
    print ("flattened2={}".format(flattened.tag.test_value[0:38,imgpixel]))

    shuffled2 = flattened.dimshuffle((1,0))
    print ("shuffled2 shape={}".format(shuffled2.tag.test_value.shape))
    print ("shuffled2={}".format(shuffled2.tag.test_value[imgpixel,0:38]))
    '''

    if (prediction_transform is not None):
        reshaped_predictions = network_output.flatten(3).dimshuffle((1,0,2)).flatten(2).dimshuffle((1,0))
        predictions = prediction_transform(reshaped_predictions)
        #print ("Transformed output shape={}".format(predictions.tag.test_value.shape))
        target = target.flatten()
        #print ("Flattened Target shape={}".format(target.tag.test_value.shape))
    else:
        predictions = network_output


    if (weights is not None):
       weight_vals = weights[target]
       loss = aggregate(loss_function(predictions, target),weights=weight_vals, mode ='normalized_sum')
       print ("Using weights")
    else:
        print ("Not Using weights")
        loss = aggregate(loss_function(predictions, target),mode='mean')





    #print (weight_vals.tag.test_value)
    #T.inc_subtensor (weight_vals[:],weights[flattened_target],inplace=True)
    #print("Test loss ={}".format(loss.tag.test_value))


    if l1 > 0:
        loss += regularization.regularize_network_params(
            layers[-1], regularization.l1) * l1
    if l2 > 0:
        loss += regularization.regularize_network_params(
            layers[-1], regularization.l2) * l2

    #print ("With regularization ={}".format(loss.tag.test_value))
    return loss


class NeuralNet(BaseEstimator):
    """A scikit-learn estimator based on Lasagne.

    """
    def __init__(
        self,
        layers,
        update=nesterov_momentum,
        loss=None,  # BBB
        objective=objective,
        objective_loss_function=None,
        objective_weights= None,
        batch_iterator_train=BatchIterator(batch_size=128),
        batch_iterator_valid=BatchIterator(batch_size=128),
        batch_iterator_test=BatchIterator(batch_size=16),
        regression=False,
        prediction_transform=T.nnet.softmax,
        max_epochs=100,
        train_split=TrainSplit(eval_size=0.2),
        custom_scores=None,
        X_tensor_type=None,
        y_tensor_type=None,
        use_label_encoder=False,
        y_transform=None,
        on_batch_finished=None,
        on_epoch_finished=None,
        on_training_started=None,
        on_training_finished=None,
        more_params=None,
        verbose=0,
        validation_output_folder = None,
        train_output = True,
        cmap =plt.get_cmap('hsv'),
        norm = None,
        label_list = None,
        validation_plot_func = None,
	num_imgs_out = 1, 
        **kwargs
        ):
        """:param layers: A list of lasagne layers to compose into the final
                          neural net

        :param on_epoch_finished: A list of functions which are called
                                 after every epoch.  The functions
                                 will be passed the NeuralNet as the
                                 first parameter and its
                                 train_history_ attribute as the
                                 second parameter.

        :param on_training_started: A list of functions which are
                                    called after training has started.
                                    The functions will be passed the
                                    NeuralNet as the first parameter
                                    and its train_history_ attribute
                                    as the second parameter.

        :param on_training_finished: A list of functions which are
                                     called after training is
                                     finished.  The functions will be
                                     passed the NeuralNet as the first
                                     parameter and its train_history_
                                     attribute as the second
                                     parameter.

        """
        if loss is not None:
            raise ValueError(
                "The 'loss' parameter was removed, please use "
                "'objective_loss_function' instead.")  # BBB
        if hasattr(objective, 'get_loss'):
            raise ValueError(
                "The 'Objective' class is no longer supported, please "
                "use 'nolearn.lasagne.objective' or similar.")  # BBB
        if objective_loss_function is None:
            objective_loss_function = (
                squared_error if regression else categorical_crossentropy)

        if hasattr(self, 'train_test_split'):  # BBB
            warn("The 'train_test_split' method has been deprecated, please "
                 "use the 'train_split' parameter instead.")
            train_split = LegacyTrainTestSplit(
                eval_size=kwargs.pop('eval_size', 0.2))

        if 'eval_size' in kwargs:  # BBB
            warn("The 'eval_size' argument has been deprecated, please use "
                 "the 'train_split' parameter instead, e.g.\n"
                 "train_split=TrainSplit(eval_size=0.4)")
            train_split.eval_size = kwargs.pop('eval_size')

        if y_tensor_type is None:
            if regression:
                y_tensor_type = T.TensorType(
                    theano.config.floatX, (False, False))
            else:
                y_tensor_type = T.ivector

        if X_tensor_type is not None:
            raise ValueError(
                "The 'X_tensor_type' parameter has been removed. "
                "It's unnecessary.")  # BBB

        if 'custom_score' in kwargs:
            warn("The 'custom_score' argument has been deprecated, please use "
                 "the 'custom_scores' parameter instead, which is just "
                 "a list of custom scores e.g.\n"
                 "custom_scores=[('first output', lambda y1, y2: abs(y1[0,0]-y2[0,0])), ('second output', lambda y1,y2: abs(y1[0,1]-y2[0,1]))]")

            # add it to custom_scores
            if custom_scores is None:
                custom_scores = [kwargs.pop('custom_score')]
            else:
                custom_scores.append(kwargs.pop('custom_score'))

        if isinstance(layers, Layer):
            layers = _list([layers])
        self.validation_plot_func = validation_plot_func

        self.layers = layers
        self.update = update
        self.objective = objective
        self.objective_loss_function = objective_loss_function
        if (objective_weights is not None):
            self.objective_weights = T.as_tensor_variable(objective_weights)
        else:
            self.objective_weights = objective_weights

        self.batch_iterator_train = batch_iterator_train if type(batch_iterator_train) is list else [batch_iterator_train]
        self.batch_iterator_test = batch_iterator_test  if type(batch_iterator_test) is list else [batch_iterator_test]
        self.batch_iterator_valid = batch_iterator_valid  if type(batch_iterator_valid) is list else [batch_iterator_valid]
        self.regression = regression
        self.max_epochs = max_epochs
        self.train_split = train_split
        self.custom_scores = custom_scores
        self.y_tensor_type = y_tensor_type
        self.use_label_encoder = use_label_encoder
        self.y_transform = y_transform
        self.on_batch_finished = on_batch_finished or []
        self.on_epoch_finished = on_epoch_finished or []
        self.on_training_started = on_training_started or []
        self.on_training_finished = on_training_finished or []
        self.more_params = more_params or {}
        self.prediction_transform = prediction_transform
        self.verbose = verbose
        if self.verbose:
            # XXX: PrintLog should come before any other handlers,
            # because early stopping will otherwise cause the last
            # line not to be printed
            self.on_epoch_finished.append(PrintLog())
            self.on_training_started.append(PrintLayerInfo())

        for key in kwargs.keys():
            assert not hasattr(self, key)
        vars(self).update(kwargs)
        self._kwarg_keys = list(kwargs.keys())
        self.validation_output_folder = validation_output_folder
        if (self.validation_output_folder is not None) and (not os.path.exists(self.validation_output_folder)):
            os.makedirs(self.validation_output_folder)

        if (train_output == True):
            self.train_output_folder = os.path.join ( self.validation_output_folder, 'training_out')
        else:
            self.train_output_folder = None
        if (self.train_output_folder is not None) and (not os.path.exists(self.train_output_folder)):
            os.makedirs(self.train_output_folder)

        self.train_history_ = []
        self.cmap = cmap
        self.norm = norm
        self.label_list = label_list
        self.num_imgs_out = num_imgs_out
        if 'batch_iterator' in kwargs:  # BBB
            raise ValueError(
                "The 'batch_iterator' argument has been replaced. "
                "Use 'batch_iterator_train' and 'batch_iterator_test' instead."
                )

    def _check_for_unused_kwargs(self):
        names = self.layers_.keys() + ['update', 'objective']
        for k in self._kwarg_keys:
            for n in names:
                prefix = '{}_'.format(n)
                if k.startswith(prefix):
                    break
            else:
                raise ValueError("Unused kwarg: {}".format(k))

    def _check_good_input(self, X, y=None):
        if isinstance(X, dict):
            lengths = [len(X1) for X1 in X.values()]
            if len(set(lengths)) > 1:
                raise ValueError("Not all values of X are of equal length.")
            x_len = lengths[0]
        else:
            x_len = len(X)

        if y is not None:
            if len(y) != x_len:
                raise ValueError("X and y are not of equal length.")

        if self.regression and y is not None and y.ndim == 1:
            y = y.reshape(-1, 1)

        return X, y

    def initialize(self):
        if getattr(self, '_initialized', False):
            return

        out = getattr(self, '_output_layer', None)
        if out is None:
            out = self._output_layer = self.initialize_layers()
        self._check_for_unused_kwargs()

        iter_funcs = self._create_iter_funcs(
            self.layers_, self.objective, self.update,
            self.y_tensor_type,
            )
        self.train_iter_, self.eval_iter_, self.predict_iter_ = iter_funcs
        self._initialized = True

    def _get_params_for(self, name):
        collected = {}
        prefix = '{}_'.format(name)

        params = vars(self)
        more_params = self.more_params

        for key, value in itertools.chain(params.items(), more_params.items()):
            if key.startswith(prefix):
                collected[key[len(prefix):]] = value

        return collected

    def _layer_name(self, layer_class, index):
        return "{}{}".format(
            layer_class.__name__.lower().replace("layer", ""), index)

    def initialize_layers(self, layers=None):
        if layers is not None:
            self.layers = layers
        self.layers_ = Layers()

        if isinstance(self.layers[0], Layer):
            # 'self.layers[0]' is already the output layer with type
            # 'lasagne.layers.Layer', so we only have to fill
            # 'self.layers_' and we're done:
            for i, layer in enumerate(get_all_layers(self.layers[0])):
                name = layer.name or self._layer_name(layer.__class__, i)
                self.layers_[name] = layer
                if self._get_params_for(name) != {}:
                    raise ValueError(
                        "You can't use keyword params when passing a Lasagne "
                        "instance object as the 'layers' parameter of "
                        "'NeuralNet'."
                        )
                #print ("{} Layer {} output shape: {}".format(i,name,layer.output_shape))
            return self.layers[0]

        # 'self.layers' are a list of '(Layer class, kwargs)', so
        # we'll have to actually instantiate the layers given the
        # arguments:
        layer = None
        for i, layer_def in enumerate(self.layers):
            print ("Initializing {} from kwargs".format(i))
            # New format: (Layer, {'layer': 'kwargs'})
            layer_factory, layer_kw = layer_def
            layer_kw = layer_kw.copy()

            if 'name' not in layer_kw:
                layer_kw['name'] = self._layer_name(layer_factory, i)

            more_params = self._get_params_for(layer_kw['name'])
            layer_kw.update(more_params)

            if layer_kw['name'] in self.layers_:
                raise ValueError(
                    "Two layers with name {}.".format(layer_kw['name']))

            # Any layers that aren't subclasses of InputLayer are
            # assumed to require an 'incoming' paramter.  By default,
            # we'll use the previous layer as input:
            if not issubclass(layer_factory, InputLayer):
                if 'incoming' in layer_kw:
                    layer_kw['incoming'] = self.layers_[
                        layer_kw['incoming']]
                elif 'incomings' in layer_kw:
                    layer_kw['incomings'] = [
                        self.layers_[name] for name in layer_kw['incomings']]
                else:
                    layer_kw['incoming'] = layer

            if issubclass(layer_factory, InverseLayer):
                if 'layer' in layer_kw:
                    layer_kw['layer'] = self.layers_[layer_kw['layer']]
                else:
                    layer_kw['layer'] = layer

            for attr in ('W', 'b'):
                if isinstance(layer_kw.get(attr), str):
                    name = layer_kw[attr]
                    layer_kw[attr] = getattr(self.layers_[name], attr, None)

            try:
                layer_wrapper = layer_kw.pop('layer_wrapper', None)
                layer = layer_factory(**layer_kw)
            except TypeError as e:
                msg = ("Failed to instantiate {} with args {}.\n"
                       "Maybe parameter names have changed?".format(
                           layer_factory, layer_kw))
                chain_exception(TypeError(msg), e)

            self.layers_[layer_kw['name']] = layer
            if layer_wrapper is not None:
                layer = layer_wrapper(layer)
                self.layers_["LW_%s" % layer_kw['name']] = layer
            #print (layer.output_shape)
        return layer

    def _create_iter_funcs(self, layers, objective, update, output_type):
        y_batch = output_type('y_batch')

        output_layer = layers[-1]
        objective_kw = self._get_params_for('objective')

        input_layers = [layer for layer in layers.values()
                        if isinstance(layer, InputLayer)]

        loss_train = objective(
            layers, prediction_transform = self.prediction_transform, target=y_batch, **objective_kw)
        loss_eval = objective(
            layers, prediction_transform = self.prediction_transform, target=y_batch, deterministic=True, **objective_kw)
        self.loss_eval_ = loss_eval

        predict_proba = get_output(output_layer, None, deterministic=True)
        if not self.regression:
            predict = T.argmax(predict_proba,axis=1, keepdims = True)
            accuracy = T.mean(T.eq(predict, y_batch))
        else:
            accuracy = loss_eval

        all_params = self.get_all_params(trainable=True)
        grads = theano.grad(loss_train, all_params)
        for idx, param in enumerate(all_params):
            grad_scale = getattr(param.tag, 'grad_scale', 1)
            if grad_scale != 1:
                grads[idx] *= grad_scale
        update_params = self._get_params_for('update')
        updates = update(grads, all_params, **update_params)


        X_inputs = [theano.In(input_layer.input_var, name=input_layer.name)
                    for input_layer in input_layers]
        inputs = X_inputs + [theano.In(y_batch, name="y")]

        train_iter = theano.function(
            inputs=inputs,
            outputs=[loss_train, accuracy],
            updates=updates,
            allow_input_downcast=True,
            )
        eval_iter = theano.function(
            inputs=inputs,
            outputs=[loss_eval, accuracy],
            allow_input_downcast=True,
            )
        predict_iter = theano.function(
            inputs=X_inputs,
            outputs=predict_proba,
            allow_input_downcast=True,
            )

        return train_iter, eval_iter, predict_iter

    def fit(self, X, y, epochs=None):
        if (X is not None):
            X, y = self._check_good_input(X, y)

            if self.use_label_encoder:
                self.enc_ = LabelEncoder()
                y = self.enc_.fit_transform(y).astype(np.int32)
                self.classes_ = self.enc_.classes_
            self.initialize()

            if self.y_transform is not None:
                print ("Shape before:{}".format(y.shape))
                y = self.y_transform(y)
                print ("Shape after:{}".format(y.shape))

        try:
            self.train_loop(X, y, epochs=epochs)
        except KeyboardInterrupt:
            pass
        return self

    def partial_fit(self, X, y, classes=None):
        return self.fit(X, y, epochs=1)

    def train_loop(self, X, y, epochs=None):
        epochs = epochs or self.max_epochs
        if (X is not None):
            X_train, X_valid, y_train, y_valid = self.train_split(X, y, self)
        else:
            X_train = y_train = X_valid = y_valid = None

        on_batch_finished = self.on_batch_finished
        if not isinstance(on_batch_finished, (list, tuple)):
            on_batch_finished = [on_batch_finished]

        on_epoch_finished = self.on_epoch_finished
        if not isinstance(on_epoch_finished, (list, tuple)):
            on_epoch_finished = [on_epoch_finished]

        on_training_started = self.on_training_started
        if not isinstance(on_training_started, (list, tuple)):
            on_training_started = [on_training_started]

        on_training_finished = self.on_training_finished
        if not isinstance(on_training_finished, (list, tuple)):
            on_training_finished = [on_training_finished]

        epoch = 0
        best_valid_loss = (
            min([row['valid_loss'] for row in self.train_history_]) if
            self.train_history_ else np.inf
            )
        best_train_loss = (
            min([row['train_loss'] for row in self.train_history_]) if
            self.train_history_ else np.inf
            )
        
        for func in on_training_started:
            func(self, self.train_history_)

        num_epochs_past = len(self.train_history_)

        PrintLog()
        while epoch < epochs:
            epoch += 1

            train_losses = []
            train_accuracies = []
            valid_losses = []
            valid_accuracies = []
            if self.custom_scores:
                custom_scores = [[] for _ in self.custom_scores]
            else:
                custom_scores = []

            t_train = 0
            t_val = 0
            t_get_img = 0
            t_start = time()
            total_disp_count = 0
            for b_idx, batch_iterator_train in enumerate(self.batch_iterator_train):
              for idx, (Xdisp, ydisp, dt,names) in enumerate(batch_iterator_train(X_valid, y_valid,  use_norm = False)):
                  t_get_img += dt
                  t_start_train = time()
                  Xb,yb = batch_iterator_train.normalize(Xdisp.copy(),ydisp)
                  if self.y_transform is not None:
                      yb = self.y_transform(yb)
                  batch_train_loss, batch_train_accuracy = self.apply_batch_func(
                      self.train_iter_, Xb, yb)
                  train_losses.append(batch_train_loss)
                  train_accuracies.append (batch_train_accuracy)

                  if (self.train_output_folder is not None and idx < self.num_imgs_out):
                      predictions = self.predict (Xb)
                      DummySelf = namedtuple('DummySelf', ['label_list','cmap','norm'])
                      dummy_self = DummySelf (self.label_list, self.cmap,self.norm)
                      if (self.validation_plot_func is not None):
                          self.validation_plot_func(self=dummy_self, Xb=Xdisp, yb=ydisp,p_out=predictions,epoch=(num_epochs_past + epoch), start_idx=total_disp_count, output_folder= self.train_output_folder)
                          total_disp_count += Xdisp.shape[0]
                          if (idx >= self.num_imgs_out):
                              break

                  for func in on_batch_finished:
                      func(self, self.train_history_)
                  t_train += time() - t_start_train
              
            total_disp_count = 0
            for b_idx, batch_iterator_valid in enumerate(self.batch_iterator_valid):
              for idx, (Xdisp, ydisp, dt,names) in enumerate(batch_iterator_valid(X_valid, y_valid,  use_norm = False)):
                  t_get_img += dt
                  t_start_val = time()
                  Xb,yb = batch_iterator_valid.normalize(Xdisp.copy(),ydisp)
                  if self.y_transform is not None:
                      yb = self.y_transform(yb)
                  batch_valid_loss, accuracy = self.apply_batch_func(
                      self.eval_iter_, Xb, yb)
                  valid_losses.append(batch_valid_loss)
                  valid_accuracies.append(accuracy)

                  if self.custom_scores:
                      y_prob = self.apply_batch_func(self.predict_iter_, Xb)
                      for custom_scorer, custom_score in zip(self.custom_scores, custom_scores):
                          custom_score.append(custom_scorer[1](yb, y_prob))

                  if (self.validation_output_folder is not None and idx < self.num_imgs_out):
                      predictions = self.predict (Xb)
                      DummySelf = namedtuple('DummySelf', ['label_list','cmap','norm'])
                      dummy_self = DummySelf (self.label_list, self.cmap,self.norm)
                      if (self.validation_plot_func is not None):
                          self.validation_plot_func(self=dummy_self, Xb=Xdisp, yb=ydisp,p_out=predictions,epoch=(num_epochs_past + epoch), start_idx=total_disp_count, output_folder= self.validation_output_folder)
                          total_disp_count += Xdisp.shape[0]
                          if (idx >= self.num_imgs_out):
                              break

                  t_val += time() - t_start_val
            avg_train_loss = np.mean(train_losses)
            avg_valid_loss = np.mean(valid_losses)
            avg_valid_accuracy = np.mean(valid_accuracies)
            avg_train_accuracy = np.mean(train_accuracies)
            if custom_scores:
                avg_custom_scores = np.mean(custom_scores, axis=1)

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss


            try: 
                '''
                if (self.validation_output_folder is not None):
                    if (epoch % 1 == 0):
                        MAX_DISPLAY = 8
                        predictions = []
                        for idx, (Xdisp, ydisp,dt,names) in enumerate(self.batch_iterator_valid(X_valid, y_valid, use_norm = False)):
                            Xb,yb = self.batch_iterator_valid.normalize(Xdisp.copy(),ydisp)
                            predictions = self.predict (Xb)

                            DummySelf = namedtuple('DummySelf', ['label_list','cmap','norm'])
                            dummy_self = DummySelf (self.label_list, self.cmap,self.norm)
                            if (self.validation_plot_func is not None):
                                self.validation_plot_func(self=dummy_self, Xb=Xdisp, yb=ydisp,p_out=predictions,epoch=(num_epochs_past + epoch), start_idx=idx, output_folder= self.validation_output_folder)
                                if (idx >= MAX_DISPLAY):
                                    break
                '''
                

                t_total = time() - t_start

                info = {
                    'epoch': num_epochs_past + epoch,
                    'train_loss': avg_train_loss,
                    'train_loss_best': best_train_loss == avg_train_loss,
                    'valid_loss': avg_valid_loss,
                    'valid_loss_best': best_valid_loss == avg_valid_loss,
                    'valid_accuracy': avg_valid_accuracy,
                    'train_accuracy': avg_train_accuracy,
                    'train_dur': t_train,
                    'val_dur': t_val,
                    'img_dur': t_get_img,
                    'total_dur': t_total
                    }
                if self.custom_scores:
                    for index, custom_score in enumerate(self.custom_scores):
                        info[custom_score[0]] = avg_custom_scores[index]
                self.train_history_.append(info)
                for func in on_epoch_finished:
                    func(self, self.train_history_)

                if (self.validation_output_folder is not None):
                    filename = os.path.join(self.validation_output_folder, 'epoch_{:04d}'.format(num_epochs_past + epoch))
                    self.write_model_data_hdf5 (filename)

            except StopIteration:
                break



        for func in on_training_finished:
            func(self, self.train_history_)

    @staticmethod
    def apply_batch_func(func, Xb, yb=None):
        if isinstance(Xb, dict):
            kwargs = dict(Xb)
            if yb is not None:
                kwargs['y'] = yb
            return func(**kwargs)
        else:
            return func(Xb) if yb is None else func(Xb, yb)

    def predict_proba(self, X):
        probas = []
        for batch_iterator_test in self.batch_iterator_test:
            for Xb, yb in batch_iterator_test(X):
                probas.append(self.apply_batch_func(self.predict_iter_, Xb))
        return np.vstack(probas)

    def predict(self, X):
        #if self.regression:
        return self.predict_proba(X)
        #else:
    #        y_pred = np.argmax(self.predict_proba(X), axis=1)
    #        if self.use_label_encoder:
    #            y_pred = self.enc_.inverse_transform(y_pred)
    #        return y_pred

    def get_output(self, layer, X):
        if isinstance(layer, basestring):
            layer = self.layers_[layer]

        fn_cache = getattr(self, '_get_output_fn_cache', None)
        if fn_cache is None:
            fn_cache = {}
            self._get_output_fn_cache = fn_cache

        if layer not in fn_cache:
            xs = self.layers_[0].input_var.type()
            get_activity = theano.function([xs], get_output(layer, xs))
            fn_cache[layer] = get_activity
        else:
            get_activity = fn_cache[layer]

        outputs = []
        for batch_iterator_test in self.batch_iterator_test:
            for Xb, yb in batch_iterator_test(X):
                outputs.append(get_activity(Xb))
        return np.vstack(outputs)

    def score(self, X, y):
        score = mean_squared_error if self.regression else accuracy_score
        return float(score(self.predict(X), y))

    def get_all_layers(self):
        return self.layers_.values()

    def get_all_params(self, **kwargs):
        layers = self.get_all_layers()
        params = sum([l.get_params(**kwargs) for l in layers], [])
        return unique(params)

    def get_all_params_values(self):
        return_value = OrderedDict()
        for name, layer in self.layers_.items():
            return_value[name] = [p.get_value() for p in layer.get_params()]
        return return_value

    def load_params_from(self, source):
        self.initialize()

        if isinstance(source, basestring):
            with open(source, 'rb') as f:
                source = pickle.load(f)

        if isinstance(source, NeuralNet):
            source = source.get_all_params_values()

        success = "Loaded parameters to layer '{}' (shape {})."
        failure = ("Could not load parameters to layer '{}' because "
                   "shapes did not match: {} vs {}.")

        for key, values in source.items():
            layer = self.layers_.get(key)
            if layer is not None:
                for p1, p2v in zip(layer.get_params(), values):
                    shape1 = p1.get_value().shape
                    shape2 = p2v.shape
                    shape1s = 'x'.join(map(str, shape1))
                    shape2s = 'x'.join(map(str, shape2))
                    if shape1 == shape2:
                        p1.set_value(p2v)
                        if self.verbose:
                            print(success.format(
                                key, shape1s, shape2s))
                    else:
                        if self.verbose:
                            print(failure.format(
                                key, shape1s, shape2s))

    def save_params_to(self, fname):
        params = self.get_all_params_values()
        with open(fname, 'wb') as f:
            pickle.dump(params, f, -1)

    def load_weights_from(self, source):
        warn("The 'load_weights_from' method will be removed in nolearn 0.6. "
             "Please use 'load_params_from' instead.")

        if isinstance(source, list):
            raise ValueError(
                "Loading weights from a list of parameter values is no "
                "longer supported.  Please send me something like the "
                "return value of 'net.get_all_params_values()' instead.")

        return self.load_params_from(source)

    def save_weights_to(self, fname):
        warn("The 'save_weights_to' method will be removed in nolearn 0.6. "
             "Please use 'save_params_to' instead.")
        return self.save_params_to(fname)

    def __getstate__(self):
        state = dict(self.__dict__)
        for attr in (
            'train_iter_',
            'eval_iter_',
            'predict_iter_',
            '_initialized',
            '_get_output_fn_cache',
            ):
            if attr in state:
                del state[attr]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.initialize()

    def get_params(self, deep=True):
        params = super(NeuralNet, self).get_params(deep=deep)

        # Incidentally, Lasagne layers have a 'get_params' too, which
        # for sklearn's 'clone' means it would treat it in a special
        # way when cloning.  Wrapping the list of layers in a custom
        # list type does the trick here, but of course it's crazy:
        params['layers'] = _list(params['layers'])
        return _dict(params)

    def _get_param_names(self):
        # This allows us to have **kwargs in __init__ (woot!):
        param_names = super(NeuralNet, self)._get_param_names()
        return param_names + self._kwarg_keys


    def read_model_data(self, filename):
        """Unpickles and loads parameters into a Lasagne model."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        lasagne.layers.set_all_param_values(self.layers_[-1], data)


    def write_model_data(self, filename):
        """Pickels the parameters within a Lasagne model."""
        data = lasagne.layers.get_all_param_values(self.layers_[-1])
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def write_model_data_hdf5 (self, filename):
        data = lasagne.layers.get_all_param_values(self.layers_[-1])
        with h5py.File(filename+".hdf5","w") as h5file:
            counter = 0
            h5file.attrs["num_layers"] = len(data)
            group = h5file.create_group("Layers")
            for layer in data:
              group.create_dataset("{:06d}".format(counter), data=layer,compression="lzf")
              counter += 1
            group = h5file.create_group("train_history")
            for i,info in enumerate(self.train_history_):
              info_group = group.create_group("{:04d}".format(i+1))
              for key,val in info.items():
                  info_group.create_dataset(key, data=val)


    def read_model_data_hdf5 (self, filename):
        read_data =[]
        train_history = []
        with h5py.File(filename,"r") as h5file:
            num_layers = h5file.attrs["num_layers"]
            group = h5file['Layers']
            for counter in range(num_layers):
              layer = group["{:06d}".format(counter)][:]
              read_data.append(layer)
            if ('train_history' in h5file):
                group = h5file["train_history"]
                info_groups = list (group)
                train_history = [{}] * len(info_groups)
                for info in info_groups:
                  epoch_num = int(info)
                  info = group[info]
                  temp_dic = {}
                  for key,val in info.items():
                      temp_dic[key] = val[()]
                  train_history [epoch_num-1] = temp_dic
        self.train_history_ = train_history
        lasagne.layers.set_all_param_values(self.layers_[-1], read_data)
