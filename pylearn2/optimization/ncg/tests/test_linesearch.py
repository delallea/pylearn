import numpy, numpy as np
import theano.tensor as TT, theano
import rnn.gd_methods.linesearch_module as lsrc
from python_cg import *
import time

zero = TT.constant(numpy.zeros((1,), dtype = theano.config.floatX))

# Global parameters
floatX      = theano.config.floatX
profile     = True
pydot_print = False


def test_scalar_search():
    W = TT.shared(numpy.asarray(
            numpy.random.uniform(size=(784,500)),
            dtype = floatX), name ='W')
    v = TT.shared(numpy.asarray(
            numpy.random.uniform(size=(40,784)),
            dtype = floatX), name = 'v')

    def phi(x):
        return TT.tanh(TT.dot(x*v,W)).sum()

    def derphi(x):
        return TT.grad( TT.tanh(TT.dot(x*v,W)).sum(), x),[zero]


    x = TT.scalar('x')

    func  = theano.function([x], phi(x), allow_input_downcast = True)
    gfunc = theano.function([x], derphi(x)[0], allow_input_downcast = True)



    #f = func
    #gf = gfunc
    #phi_calls = 0
    #derphi_calls = 0

    g_a_lo = numpy.array( (numpy.random.uniform( size = (50,))), dtype =floatX)
    g_a_hi = numpy.array( (numpy.random.uniform( size = (50,))), dtype =floatX)
    old_phi0 = theano.shared(numpy.asarray(func(0), dtype=floatX),
                               name = 'old_phi0')
    phi0 = theano.shared(numpy.asarray(func(0), dtype=floatX),
                               name = 'phi0')
    derphi0 = theano.shared(numpy.asarray(func(0), dtype=floatX),
                                  name = 'derphi0')
    a_lo = theano.shared(numpy.asarray(0.2, dtype=floatX),
                                name ='a_lo')
    a_hi = theano.shared(numpy.asarray(0.8, dtype=floatX),
                               name = 'a_hi')
    phi_lo = theano.shared(numpy.asarray(func(0.2),
                                                dtype=floatX),
                                 name = 'phi_lo')
    phi_hi = theano.shared(numpy.asarray(func(0.8),
                                                dtype = floatX),
                                 name = 'phi_hi')
    derphi_lo = theano.shared(numpy.asarray(gfunc(0.2),
                                                   dtype=floatX),
                                    name = 'derphi_lo')
    c1 = 1e-4
    c2 = 0.9

    outs = lsrc.scalar_search_wolfe2(phi, derphi, profile = profile)

    f = theano.function([], list(outs[:-1])+list(outs[-1]), profile = profile,
                       name = 'test_scalar_search',
                       mode = theano.Mode(linker='cvm'))


    #if pydot_print:
    #    theano.plot.Ipydotprint( f, 'non_frugal')

    theano.printing.pydotprint(f, 'func.png', scan_graphs=True, with_ids =
                               True)
    t_th = 0
    for k in xrange(1):
        t0 = time.time()
        thrval = f()
        t_th += time.time() - t0

    phi_calls, f     = wrap_function(func, [])
    derphi_calls, gf = wrap_function(gfunc, [])
    t_py = 0
    for k in xrange(1):
        t0 = time.time()
        rval = py_scalar_search_wolfe2(f, gf)
        t_py += time.time() - t0

    print 'NCALLs phi', phi_calls
    print 'NCALLs derphi', derphi_calls
    print 'THEANO output :: ',thrval
    print 'NUMPY  output :: ',rval
    print
    print 'Timings'
    print
    print 'theano ---------> time %e'% t_th
    print 'numpy  ---------> time %e'% t_py

def test_zoom():


    W = TT.shared(numpy.asarray(
            numpy.random.uniform(size=(784,500)),
            dtype = floatX), name ='W')
    v = TT.shared(numpy.asarray(
            numpy.random.uniform(size=(40,784)),
            dtype = floatX), name = 'v')

    def phi(x):
        return TT.tanh(TT.dot(x*v,W)).sum()

    def derphi(x):
        return TT.grad( TT.tanh(TT.dot(x*v,W)).sum(), x),[zero]


    x = TT.scalar('x')
    func  = theano.function([x], phi(x), allow_input_downcast = True)
    gfunc = theano.function([x], derphi(x)[0], allow_input_downcast = True)

    old_phi0 = theano.shared(numpy.asarray(func(0), dtype=floatX),
                               name = 'old_phi0')
    phi0 = theano.shared(numpy.asarray(func(0), dtype=floatX),
                               name = 'phi0')
    derphi0 = theano.shared(numpy.asarray(func(0), dtype=floatX),
                                  name = 'derphi0')

    g_a_lo = numpy.array( (numpy.random.uniform( size = (50,))), dtype =floatX)
    g_a_hi = numpy.array( (numpy.random.uniform( size = (50,))), dtype =floatX)
    a_lo = theano.shared(numpy.asarray(0.2, dtype=floatX),
                                name ='a_lo')
    a_hi = theano.shared(numpy.asarray(0.8, dtype=floatX),
                               name = 'a_hi')
    phi_lo = theano.shared(numpy.asarray(func(0.2),
                                                dtype=floatX),
                                 name = 'phi_lo')
    phi_hi = theano.shared(numpy.asarray(func(0.8),
                                                dtype = floatX),
                                 name = 'phi_hi')
    derphi_lo = theano.shared(numpy.asarray(gfunc(0.2),
                                                   dtype=floatX),
                                    name = 'derphi_lo')
    c1 = 1e-4
    c2 = 0.9

    outs = lsrc._zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo, phi, derphi, phi0,
                 derphi0, c1,c2,[zero], profile)


    f = theano.function([], list(outs[:-1])+list(outs[-1]), profile = profile,
                        name = 'test_zoom', mode = theano.Mode(linker='cvm'))
    theano.printing.pydotprint(f, 'func.png', scan_graphs = True)

    if pydot_print:
       theano.plot.Ipydotprint( f, 'non_frugal')

    t_th = 0
    for k in xrange(10):
        a_lo.set_value(g_a_lo[k])
        a_hi.set_value(g_a_hi[k])
        phi_lo.set_value(func(g_a_lo[k]))
        phi_hi.set_value(func(g_a_hi[k]))
        derphi_lo.set_value(gfunc(g_a_lo[k]))
        t0 = time.time()
        thrval = f()
        t_th += time.time() - t0

    phi_calls, f     = wrap_function(func, [])
    derphi_calls, gf = wrap_function(gfunc, [])
    t_py = 0
    for k in xrange(10):
        t0 = time.time()
        rval = py_zoom(g_a_lo[k],g_a_hi[k], func(g_a_lo[k]),
                       func(g_a_hi[k]), gfunc(g_a_lo[k]), f, gf,
                  func(0), gfunc(0), c1,c2)
        t_py += time.time() - t0

    print 'NCALLs phi', phi_calls
    print 'NCALLs derphi', derphi_calls
    print 'THEANO output :: ',thrval
    print 'NUMPY  output :: ',rval
    print
    print 'Timings'
    print
    print 'theano ---------> time %e'% t_th
    print 'numpy   ---------> time %e'% t_py



def test_quadmin():
    c   = theano.shared(numpy.array(1., dtype=floatX), name = 'c')
    fc  = theano.shared(numpy.array(1., dtype=floatX), name = 'fc')
    a   = theano.shared(numpy.array(1., dtype=floatX), name = 'a')
    fa  = theano.shared(numpy.array(1., dtype=floatX), name = 'fa')
    b   = theano.shared(numpy.array(1., dtype=floatX), name = 'b')
    fb  = theano.shared(numpy.array(1., dtype=floatX), name = 'fb')
    fpa = theano.shared(numpy.array(1., dtype=floatX), name = 'fpa')

    xmin = lsrc._quadmin(a,fa,fpa,b,fb)
    f = theano.function([], xmin)

    if pydot_print:
        theano.printing.pydotprint(f, 'test_quadmin.png',
                                   with_ids      = True,
                                   high_contrast = True,
                                   scan_graphs   = True)
    t_th = 0
    t_py = 0
    for x in xrange(50):
        v = numpy.asarray(numpy.random.uniform(size=(7,)),dtype=floatX)
        # set values
        a.set_value(v[0])
        fa.set_value(v[1])
        fpa.set_value(v[2])
        b.set_value(v[3])
        fb.set_value(v[4])
        c.set_value(v[5])
        fc.set_value(v[6])


        t0 = time.time()
        th_val =  f()
        t_th += time.time() - t0
        t0 = time.time()
        py_val = py_quadmin(v[0],v[1],v[2],v[3],v[4])
        t_py += time.time() - t0
        if x % 10 == 0:
            print 'THEANO output :: ',th_val
            print 'NUMPY  output :: ',py_val

    print 'Timings'
    print
    print 'theano ---------> time %e'% t_th
    print 'numpy  ---------> time %e'% t_py

def test_cubicmin():
    c   = theano.shared(numpy.array(1., dtype=floatX), name = 'c')
    fc  = theano.shared(numpy.array(1., dtype=floatX), name = 'fc')
    a   = theano.shared(numpy.array(1., dtype=floatX), name = 'a')
    fa  = theano.shared(numpy.array(1., dtype=floatX), name = 'fa')
    b   = theano.shared(numpy.array(1., dtype=floatX), name = 'b')
    fb  = theano.shared(numpy.array(1., dtype=floatX), name = 'fb')
    fpa = theano.shared(numpy.array(1., dtype=floatX), name = 'fpa')

    xmin = lsrc._cubicmin(a,fa,fpa,b,fb,c,fc)
    f = theano.function([], xmin)
    if pydot_print:
        theano.printing.pydotprint(f, 'test_cubicmin.png',
                                   with_ids      = True,
                                   high_contrast = True,
                                   scan_graphs   = True)
    t_th = 0
    t_py = 0
    for x in xrange(50):
        v = numpy.asarray(numpy.random.uniform(size=(7,)), dtype=floatX)
        # set values
        a.set_value(v[0])
        fa.set_value(v[1])
        fpa.set_value(v[2])
        b.set_value(v[3])
        fb.set_value(v[4])
        c.set_value(v[5])
        fc.set_value(v[6])


        t0 = time.time()
        th_val =  f()
        t_th += time.time() - t0
        t0 = time.time()
        py_val = py_cubicmin(v[0],v[1],v[2],v[3],v[4],v[5],v[6])
        t_py += time.time() - t0
        if x % 10 == 0:
            print 'THEANO output :: ',th_val
            print 'NUMPY  output :: ',py_val

    print 'Timings'
    print
    print 'theano ---------> time %e'% t_th
    print 'numpy   ---------> time %e'% t_py




def test_line_search_wolfe2():

    v_v = numpy.asarray(numpy.random.uniform(size=(300000,)), dtype=floatX)
    v_x = numpy.asarray(numpy.random.uniform(size=(500,100)),  dtype = floatX)
    W = theano.shared( v_v[:250000].reshape(500,500), name='W')
    b = theano.shared( v_v[250000:].reshape(500,100), name='b')
    x = theano.shared( v_x, name='x')

    _W = TT.matrix()
    _b = TT.vector()
    def local_f(W,b,x):
        return TT.tanh(TT.dot(W,x)+b).sum()

    out = local_f(_W,_b,x)


    def f(lW,lb):
        return TT.tanh(TT.dot(lW,x)+lb).sum()

    #gparams = TT.grad(out, [_W,_b])
    def fprime(lW,lb):
        return TT.grad( TT.tanh(TT.dot(lW,x)+lb).sum(), [lW,lb])


    xks = [W,b]
    pks = TT.grad(f(W,b), xks)
    args = []
    outs = lsrc.line_search_wolfe2(f, fprime, xks, pks, args = args, gfks =
                                  pks, profile = profile)
    # import ipdb; ipdb.set_trace()
    tf = theano.function([], list(outs[:-1])+list(outs[-1]), profile =
                         profile, mode = theano.Mode(linker='cvm'))
    t_th = 0
    for k in xrange(1):
        t0 = time.time()
        t_rval = tf()
        t_th += time.time() - t0

    t_py = 0

    v     = TT.vector('v')
    x     = TT.matrix('x')
    nW    = v[:250000].reshape((500,500))
    nb    = v[250000:].reshape((500,100))
    fout  = local_f(nW, nb, x)
    func  = theano.function([v,x], fout)
    gv    = TT.grad(fout, v)
    gfunc = theano.function([v,x], gv)
    gk    = gfunc(v_v, v_x)

    for k in xrange(1):
        t0 = time.time()
        rval = py_line_search_wolfe2(func, gfunc, v_v, gk, gfk = gk,
                                     args=(v_x,))
        t_py += time.time() - t0

    print 'THEANO output :: ',t_rval[:-2]
    print 'NUMPY  output :: ',rval[0], rval[3], rval[4], rval[5]
    print
    print 'Timings'
    print
    print 'theano ---------> time %e'% t_th
    print 'numpy  ---------> time %e'% t_py


if __name__=='__main__':
    print 'TEST quadmin'
    test_quadmin()
    print
    print
    print
    print 'TEST cubicmin'
    test_cubicmin()
    print
    print
    print
    print 'TEST zoom'
    test_zoom()
    print
    print
    print
    print 'TEST scalar search'
    test_scalar_search()
    print
    print
    print
    print 'TEST search wolfe2'
    test_line_search_wolfe2()
