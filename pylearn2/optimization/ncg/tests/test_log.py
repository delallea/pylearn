import numpy, numpy as np
import theano.tensor as TT, theano
from rnn.gd_methods import ncg
from python_cg import *
import time
import cPickle, gzip




# Global parameters
floatX      = theano.config.floatX
profile     = True
pydot_print = True


def test_ncg():

    rng = numpy.random.RandomState(232)
    n_in  = 784
    n_out = 10


    all_vals = numpy.zeros(n_in*n_out+n_out, dtype = theano.config.floatX)

    idx  = 0
    vW   = all_vals[0:n_in*n_out].reshape((n_in,n_out))
    vb   = all_vals[n_in*n_out:n_in*n_out+n_out]

    W    = theano.shared(vW, 'W')
    b    = theano.shared(vb, 'b')


    f = gzip.open('mnist.pkl.gz','rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, TT.cast(shared_y, 'int32')


    test_set_x,  test_set_y  = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)


    def f(W,b):
        return TT.mean((TT.dot(TT.nnet.sigmoid(TT.dot(test_set_x,W)+b),W.T) -
                test_set_x)**2)

        #p_y_given_x = TT.nnet.softmax(TT.dot(train_set_x, W)+b)
        return -TT.mean(
            TT.log(
                p_y_given_x)[TT.arange(train_set_y.shape[0]),train_set_y])

    print 'Executing ncg'
    print '>>> Generating Graph'
    t0 = time.time()
    answers = ncg(f, [W,b], [], maxiter = 20,
                      profile = profile)
    tf = time.time() - t0

    print 'It took', tf, 'sec'
    print '>>> Compiling graph'
    t0 = time.time()
    func = theano.function([], answers, profile = profile,
                           name = 'logreg',
                           mode = theano.Mode(linker='cvm'))
    tf = time.time() - t0
    print 'It took', tf, 'sec'
    if pydot_print:
        print '>>> Plotting graph'
        #import ipdb; ipdb.set_trace()
        #cPickle.dump(func, open('func_dump.pkl','wb'), protocol=-1)
    theano.printing.pydotprint(func,'logreg.png',
                           with_ids = True,
                           high_contrast = True,
                           scan_graphs = True)
    print 'Optimizing'
    t_th = 0
    t_py = 0
    for k in xrange(1):
        t0 = time.time()
        th_rval = func()[0]
        t_th += time.time() - t0

    print '-------- NOW SCIPY RESULTS ------'
    allw = TT.vector('all')
    #allw  = TT.specify_shape(_allw, all_vals.shape)
    idx  = 0
    W = allw[0:n_in*n_out].reshape((n_in, n_out))
    b = allw[n_in*n_out:n_in*n_out+n_out]

    out    = f(W,b)
    func   = theano.function([allw], out, profile = 1, name = 'func',
                           mode = theano.Mode(linker='cvm'))
    gall   = TT.grad(out, allw)
    fprime = theano.function([allw], gall, profile = 1, name = 'gall',
                           mode = theano.Mode(linker='cvm'))

    if pydot_print:
        theano.printing.pydotprint(func,  't2_f.png', with_ids = True,
                                   high_contrast = True)
        theano.printing.pydotprint(fprime,'t2_fprime.png', with_ids = True,
                                   high_contrast = True)

    # FIRST RUN with full_output to get an idea of how many steps where done
    t0 = time.time()
    rval = py_fmin_cg(func, all_vals, fprime = fprime,
                                  maxiter = 20,
                                   full_output = 1,
                                    disp = 1)[1]

    t_py += time.time() - t0
    # rest runs with full_output 0

    print 'THEANO output :: ',th_rval
    print 'NUMPY  output :: ',rval
    print
    print 'Timings'
    print
    print 'theano ---------> time %e'% t_th
    print 'numpy  ---------> time %e'% t_py




if __name__ == '__main__':
    print
    print
    print
    print 'Test :'
    test_ncg()
