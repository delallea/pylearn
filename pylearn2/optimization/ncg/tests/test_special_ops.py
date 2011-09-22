import numpy, numpy as np
import theano.tensor as TT, theano
from rnn.gd_methods import ncg
from python_cg import *
import time
import cPickle


# Global parameters
floatX      = theano.config.floatX
profile     = True
pydot_print = True


class Backward(theano.gof.Op):
    def __str__(self):
        return 'BACWARD PASS'

    def __hash__(self):
        return hash(type(self))

    def __eq__(self, other):
        return type(self) == type(other)

    def make_node(self, cost, x, W, y):
        return theano.gof.Apply(self, [cost, x,W,y], [x.type(), W.type(), y.type()])


    def perform(self,  node, ins, outs):
        cost,x,W,y = ins
        gx,gW,gy = outs

        gy[0] = theano._asarray(2*(numpy.dot(x,W) - y),
                             dtype = theano.config.floatX)
        gW[0] = theano._asarray(numpy.dot(2*(numpy.dot(x,W) -y),x.T),
                             dtype = theano.config.floatX)
        gx[0] = theano._asarray(numpy.dot(2*(numpy.dot(x,W) -y), W.T),
                                dtype = theano.config.floatX)

backward = Backward()


class Forward(theano.gof.Op):
    def __str__(self):
        return 'FORWARD PASS'

    def __hash__(self):
        return hash(type(self))

    def __eq__(self, other):
        return type(self) == type(other)

    def make_node(self, x, W, y):
        return theano.gof.Apply(self, [x,W,y], [TT.scalar()])

    def grad(self, inps, grads):
        out = self(*inps)
        x,W,y = inps
        return backward(out, x,W, y)

    def perform(self,  node, ins, out_):
        x,W,y = ins
        out, = out_
        out[0] = theano._asarray(numpy.sum((numpy.dot(x,W) - y)**2),
                                 dtype = theano.config.floatX)

forward = Forward()

def test_special():

    rng = numpy.random.RandomState(232)

    all_vals = numpy.asarray(
        rng.uniform(size=(500*500,)),
        dtype=theano.config.floatX)

    idx  = 0
    vW0  = all_vals.reshape((500,500))

    vx  = numpy.asarray(
        rng.uniform(size=(500,500)), dtype=theano.config.floatX)
    vy  = numpy.asarray(
        rng.uniform(size=(500,500)), dtype=theano.config.floatX)

    W0 = theano.shared(vW0, 'W0')
    x = theano.shared(vx, 'x')
    y = theano.shared(vy, 'y')
    def f(W0):
        return forward(x,W0,y)

    print 'Executing ncg'
    print '>>> Generating Graph'
    t0 = time.time()
    answers = ncg(f, [W0], [], maxiter = 6,
                      profile = profile)
    tf = time.time() - t0

    print 'It took', tf, 'sec'
    print '>>> Compiling graph'
    t0 = time.time()
    func = theano.function([], answers, profile = profile,
                           name = 'test_fmincg_2',
                          mode = theano.Mode(linker='cvm'))
    tf = time.time() - t0
    print 'It took', tf, 'sec'
    if pydot_print:
        print '>>> Plotting graph'
        #cPickle.dump(func, open('func_dump.pkl','wb'), protocol=-1)
    theano.printing.pydotprint(func,'special.png',
                           with_ids = True,
                           high_contrast = True,
                           scan_graphs = True)
    import plot
    print 'Optimizing'
    t_th = 0
    t_py = 0
    for k in xrange(1):
        t0 = time.time()
        th_rval = func()[0]
        t_th += time.time() - t0

    print '-------- NOW SCIPY RESULTS ------'
    allw = TT.vector('all')
    idx  = 0
    W0   = allw.reshape((500,500))
    out    = f(W0)
    func   = theano.function([allw], out)
    gall   = TT.grad(out, allw)
    fprime = theano.function([allw], gall)

    if pydot_print:
        theano.printing.pydotprint(func,  't2_f.png', with_ids = True,
                                   high_contrast = True)
        theano.printing.pydotprint(fprime,'t2_fprime.png', with_ids = True,
                                   high_contrast = True)
    # FIRST RUN with full_output to get an idea of how many steps where done
    t0 = time.time()
    rval = py_fmin_cg(func, all_vals, fprime = fprime,
                                  maxiter = 6,
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
    print 'Test special:'
    test_special()
