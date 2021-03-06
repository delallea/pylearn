"""
Tests for the pylearn2 autoencoder module.
"""
import numpy as np
from numpy.testing.utils import assert_allclose
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from pylearn2.autoencoder import Autoencoder


def test_autoencoder_logistic_linear_tied():
    data = np.random.randn(10, 5)
    ae = Autoencoder(5, 7, act_enc='sigmoid', act_dec='linear',
                     tied_weights=True)
    w = ae.weights.get_value()
    ae.hidbias.set_value(np.random.randn(7))
    hb = ae.hidbias.get_value()
    ae.visbias.set_value(np.random.randn(5))
    vb = ae.visbias.get_value()
    d = tensor.matrix()
    result = np.dot(1. / (1 + np.exp(-hb - np.dot(data,  w))), w.T) + vb
    ff = theano.function([d], ae.reconstruct(d))
    assert_allclose(ff(data), result)


def test_autoencoder_tanh_cos_untied():
    data = np.random.randn(10, 5)
    ae = Autoencoder(5, 7, act_enc='tanh', act_dec='cos',
                     tied_weights=False)
    w = ae.weights.get_value()
    w_prime = ae.w_prime.get_value()
    ae.hidbias.set_value(np.random.randn(7))
    hb = ae.hidbias.get_value()
    ae.visbias.set_value(np.random.randn(5))
    vb = ae.visbias.get_value()
    d = tensor.matrix()
    result = np.cos(np.dot(np.tanh(hb + np.dot(data,  w)), w_prime) + vb)
    ff = theano.function([d], ae.reconstruct(d))
    assert_allclose(ff(data), result)
