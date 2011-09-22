"""
 Stochastic Gradient Descent - in Theano
"""
__authors__ = "Razvan Pascanu"
__copyright__ = "(c) 2011, Universite de Montreal"
__license__ = "BSD"
__maintainer__ = "Razvan Pascanu"
__email__ = "r.pascanu@gmail"

import theano
import theano.tensor as TT
import numpy

my0 = TT.constant(numpy.array(0.0,dtype=theano.config.floatX))
my1 = TT.constant(numpy.array(1.0,dtype=theano.config.floatX))


def sgd(parameters,cost = None
        , gradients=None
        , updates=None
        , lr=2e-2
        , consider_constant = None
        , momentum = None
        , **kwargs):
    """
    Returns un update dictionary and a list of outputs. The update
    dictionary contains the updates rules for all parameters.

    Parameters
    ----------
    parameters: list or theano tensor
        List of parameters with respect to which SGD should be applied
    cost : theano expresion (optional)
        Cost with respect to which the gradient is computed
    gradients : list or theano tensor (optional)
        List of gradients corresponding to parameters
    updates: Updates object (theano) (optional)
        Object containing previous updates to which new ones can be added
    lr : float
        Learning rate for the provided parameters
    consider_constant : list of theano expressions (optional)
        Expresions through which SGD should not compute gradients
    momentum : float [0..1] (optional)
        Ammount of momentum used in training
    kwargs : dict of optional arguments
        Considered other arguments :
            * `scale` (float/theano expression) used to rescale the graidents
    """

    if not isinstance(parameters, (list,tuple)):
        parameters = [parameters]
    else:
        parameters = list(parameters)

    if consider_constant is None:
        consider_constant = []

    if gradients == None:
        gradients = TT.grad(cost,parameters,consider_constant = consider_constant)
    else:
        if not isinstance(gradients, (list,tuple)):
            gradients = [gradients]
        else:
            gradients = list(gradients)
        assert len(gradients) == len(parameters)
        assert numpy.all([ x.type == y.type for (x,y) in zip(gradients,
                                                             parameters)])


    if updates==None:
        updates = theano.Updates()
    for param,grad in zip(parameters,gradients):
        scale = my1
        if 'scale' in kwargs:
            print 'scaling the lr'
            scale = kwargs['scale']
        if momentum != None:
            mparam = theano.shared(param.get_value()*0.)
            updates[param] = param - scale * lr * mparam
            updates[mparam] = mparam*momentum + (1.-momentum)*grad
        else:
            updates[param] =  param - scale*lr * grad

    return updates, []

