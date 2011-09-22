"""
 Linesearch - in Theano
"""
__authors__ = "Razvan Pascanu"
__copyright__ = "(c) 2011, Universite de Montreal"
__license__ = "BSD"
__maintainer__ = "Razvan Pascanu"
__email__ = "r.pascanu@gmail"

import numpy
import theano.tensor as TT, theano

from theano.lazycond import ifelse
from theano.scan_module import until

zero = TT.constant(numpy.asarray(0, dtype = theano.config.floatX))
one  = TT.constant(numpy.asarray(1, dtype = theano.config.floatX))
nan  = TT.constant(numpy.asarray(numpy.nan, dtype= theano.config.floatX))

true = TT.constant(numpy.asarray(1, dtype = 'int8'))
false = TT.constant(numpy.asarray(0, dtype = 'int8'))

def lazy_or(name = 'none',*args):
    def apply_me(args):
        if len(args) == 1:
            return args[0]
        else:
            rval = ifelse(args[0], true, apply_me(args[1:]),
                          name = name + str(len(args)))
            return rval
    return apply_me(args)

def lazy_and(name = 'node',*args):
    def apply_me(args):
        if len(args) == 1:
            return args[0]
        else:
            rval = ifelse(TT.eq(args[0],zero), false, apply_me(args[1:]),
                         name = name + str(len(args)))
            return rval
    return apply_me(args)

def my_not(arg):
    return TT.eq(arg, zero)


def line_search_wolfe2(f, fprime, xks, pks, old_fval = None,
                       old_old_fval=None, args=(), c1=1e-4, c2=0.9,
                      profile = False, gfks = None):
    """Find alpha that satisfies strong Wolfe conditions.

    TODO: WRITE ME

    Returns
    -------
    alpha_star : tensor.scalar
        Alpha for which ``x_new = x0 + alpha * pk``.

    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions.  See Wright and Nocedal, 'Numerical Optimization',
    1999, pg. 59-60.

    For the zoom phase it uses an algorithm by [...].

    """
    c1 = TT.as_tensor_variable(c1)
    c2 = TT.as_tensor_variable(c2)

    def phi(alpha):
        new_args = [xk+pk*alpha for xk, pk in zip(xks,pks)]
        rval = f(*new_args)
        return rval

    def derphi(alpha):
        new_args = [xk + pk*alpha for xk,pk in zip(xks,pks)]
        rval =  fprime(*new_args)
        return sum([ (x*y).sum() for x,y in zip(rval, pks)]), rval
    if gfks == None:
        derphi0, gfks = derphi(zero)
    else:
        derphi0 = sum([(x*y).sum() for x,y in zip(gfks, pks)])
    rval = scalar_search_wolfe2(phi, derphi, old_fval, old_old_fval,
                                     derphi0, c1, c2,gfks,
                                     profile = profile)
    alpha_star, phi_star, old_fval, derphi_star, gfks = rval

    # WARNING: derphi_star is either NaN or the slope for gval
    return alpha_star, phi_star, old_fval, derphi_star, gfks


def scalar_search_wolfe2(phi, derphi=None, phi0=None,
                         old_phi0=None, derphi0=None,
                         c1=1e-4, c2=0.9, gfks = None,
                        profile = False):
    """Find alpha that satisfies strong Wolfe conditions.

    alpha > 0 is assumed to be a descent direction.

    Parameters
    ----------
    phi : callable f(x)
        Objective scalar function.

    derphi : callable f'(x)
        Objective function derivative (can be None)
    phi0 : float, optional
        Value of phi at s=0
    old_phi0 : float, optional
        Value of phi at previous point
    derphi0 : float, optional
        Value of derphi at s=0
    c1 : float
        Parameter for Armijo condition rule.
    c2 : float
        Parameter for curvature condition rule.
    profile : flag (boolean)
        True if you want printouts of profiling information

    Returns
    -------
    alpha_star : float
        Best alpha
    phi_star
        phi at alpha_star
    phi0
        phi at 0
    derphi_star
        derphi at alpha_star

    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions.  See Wright and Nocedal, 'Numerical Optimization',
    1999, pg. 59-60.

    For the zoom phase it uses an algorithm by [...].

    """

    if phi0 is None:
        phi0 = phi(zero)
    else:
        phi0 = TT.as_tensor_variable(phi0)

    if derphi0 is None and derphi is not None:
        derphi0, gfks = derphi(zero)
    else:
        derphi0 = TT.as_tensor_variable(derphi0)

    alpha0 = zero
    alpha0.name ='alpha0'
    if old_phi0 is not None:
        alpha1 = TT.minimum(one, numpy.asarray(1.01,
                                               dtype=theano.config.floatX)* \
                            numpy.asarray(2, dtype=theano.config.floatX)*(phi0 - old_phi0)/derphi0)
    else:
        old_phi0 = nan
        alpha1   = one

    alpha1 = TT.switch(alpha1 < zero, one, alpha1)
    alpha1.name = 'alpha1'

    # This shouldn't happen. Perhaps the increment has slipped below
    # machine precision?  For now, set the return variables skip the
    # useless while loop, and raise warnflag=2 due to possible imprecision.
    phi0 = TT.switch(TT.eq(alpha1, zero), old_phi0, phi0)
    # I need a lazyif for alpha1 == 0 !!!
    phi_a1 = ifelse(TT.eq(alpha1,zero), phi0,
                    phi(alpha1), name='phi_a1')
    phi_a1.name = 'phi_a1'

    phi_a0 = phi0
    phi_a0.name = 'phi_a0'
    derphi_a0 = derphi0
    derphi_a0.name = 'derphi_a0'
    # Make sure variables are tensors otherwise strange things happen
    c1 = TT.as_tensor_variable(c1)
    c2 = TT.as_tensor_variable(c2)
    maxiter = 10
    def while_search(alpha0, alpha1, phi_a0, phi_a1, derphi_a0, i_t, *gfks):

        derphi_a1, nw_gfks = derphi(alpha1)
        cond1 = TT.bitwise_or(phi_a1 > phi0 + c1*alpha1*derphi0,
                              TT.bitwise_and(phi_a1 >= phi_a0, i_t > zero))
        cond2 = abs(derphi_a1) <= -c2*derphi0
        cond3 = derphi_a1 >= zero
        alpha_star_c1, phi_star_c1, derphi_star_c1, gfks_c1 = \
                _zoom(alpha0, alpha1, phi_a0, phi_a1, derphi_a0,
                      phi, derphi, phi0, derphi0, c1,c2, gfks,
                     profile = profile)
        alpha_star_c3, phi_star_c3, derphi_star_c3, gfks_c3 = \
                _zoom(alpha1, alpha0, phi_a1, phi_a0, derphi_a1, phi,
                      derphi, phi0, derphi0, c1,c2, nw_gfks,
                     profile = profile)

        nw_alpha1 = alpha1 * numpy.asarray(2, dtype=theano.config.floatX)
        nw_phi = phi(nw_alpha1)
        alpha_star, phi_star, derphi_star = \
                ifelse(cond1,
                          (alpha_star_c1, phi_star_c1, derphi_star_c1),
                ifelse(cond2,
                          (alpha1, phi_a1, derphi_a1),
                ifelse(cond3,
                          (alpha_star_c3, phi_star_c3, derphi_star_c3),
                           (nw_alpha1, nw_phi, nan),
                      name = 'alphastar_c3'),
                      name = 'alphastar_c2'),
                      name ='alphastar_c1')
        rgfks = [ ifelse(cond1, x_c1,
                              ifelse(cond2, nw_x,
                              ifelse(cond3, x_c3, x)))
                 for x_c1, nw_x, x_c3, x in zip(gfks_c1, nw_gfks, gfks_c3,
                                                gfks) ]

        return ( [alpha1,
                  nw_alpha1,
                  phi_a1,
                  ifelse(lazy_or('allconds',cond1, cond2, cond3),
                         phi_a1, nw_phi, name='nwphi1'),
                  ifelse(cond1, derphi_a0, derphi_a1, name='derphi'),
                  i_t+one]+rgfks+
                  [alpha_star,
                  phi_star,
                  derphi_star],
                until(lazy_or('until_cond_',TT.eq(nw_alpha1,zero), cond1, cond2, cond3)))
    outs, updates = theano.scan(while_search,
                               outputs_info = [alpha0,
                                               alpha1,
                                               phi_a0,
                                               phi_a1,
                                               derphi_a0,
                                               zero]  #+  list(gfks)+
                                +[None for k in
                                                      list(gfks)]+
                                               [None,
                                               None,
                                               None],
                                non_sequences = list(gfks),
                               n_steps = maxiter,
                               name = 'while_search',
                               mode = theano.Mode(linker='cvm_nogc'),
                               profile = profile)
    out3 = outs[-3][-1]
    out2 = outs[-2][-1]
    out1 = outs[-1][-1]
    alpha_star, phi_star, derphi_star = \
            ifelse(TT.eq(alpha1, zero),
                        ( nan,phi0, nan),
                        ( out3, out2, out1), name = 'main_alphastar')

    rgfks = [ifelse(TT.eq(alpha1, zero), x, nw_x[-1])
             for x, nw_x in zip(gfks, outs[6:-3])]
    return alpha_star, phi_star,  phi0, derphi_star, rgfks



def _cubicmin(a,fa,fpa,b,fb,c,fc):
    """
    Finds the minimizer for a cubic polynomial that goes through the
    points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.

    If no minimizer can be found return None

    """
    # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D
    C = fpa
    D = fa
    db = b-a
    dc = c-a

    denom = (db*dc)**2 * (db-dc)
    d1_00 = dc**2
    d1_01 = -db**2
    d1_10 = -dc**3
    d1_11 = db**3
    t1_0 = fb - fa - C*db
    t1_1 = fc - fa - C*dc
    A = d1_00*t1_0 + d1_01*t1_1
    B = d1_10*t1_0 + d1_11*t1_1
    A /= denom
    B /= denom
    radical = B*B-3*A*C
    cond = lazy_or('cubicmin',radical < 0,
                         TT.eq(db , 0),
                         TT.eq(dc , 0),
                         TT.eq(b , c),
                         TT.eq(A , 0))
    # Note: `lazy if` would make more sense, but it is not
    #       implemented in C right now
    xmin = TT.switch(cond, TT.constant(numpy.nan),
                         a + (-B + TT.sqrt(radical))/(3*A))
    return xmin

def _quadmin(a,fa,fpa,b,fb):
    """
    Finds the minimizer for a quadratic polynomial that goes through
    the points (a,fa), (b,fb) with derivative at a of fpa,

    """
    # f(x) = B*(x-a)^2 + C*(x-a) + D
    D  = fa
    C  = fpa
    db = b-a*one

    B = (fb-D-C*db)/(db*db)
    # Note : `lazy if` would make more sense, but it is not
    #        implemented in C right now
    xmin = TT.switch(lazy_or('quadmin',TT.eq(db , zero), (B <= zero)),
                         TT.constant(numpy.asarray(numpy.nan,
                                                   dtype=theano.config.floatX)),
                         a - C/ (numpy.asarray(2.0,
                                               dtype=theano.config.floatX)*B) )
    return xmin

def _zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo,
          phi, derphi, phi0, derphi0, c1, c2, gfks = None,
          profile = False):
    """
    TODO: re-write me

    Part of the optimization algorithm in `scalar_search_wolfe2`.
    a_lo : scalar (step size)
    a_hi : scalar (step size)
    phi_lo : scalar (value of f at a_lo)
    phi_hi : scalar ( value of f at a_hi)
    derphi_lo : scalar ( value of derivative at a_lo)
    phi : callable -> generates computational graph
    derphi: callable -> generates computational graph
    phi0 : scalar ( value of f at 0)
    derphi0 : scalar (value of the derivative at 0)
    c1 : scalar  (wolfe parameter)
    c2 : scalar  (wolfe parameter)
    profile: if you want printouts of profiling information
    """

    a_hi      = TT.as_tensor_variable(a_hi)
    a_lo      = TT.as_tensor_variable(a_lo)
    phi_hi    = TT.as_tensor_variable(phi_hi)
    derphi_lo = TT.as_tensor_variable(derphi_lo)
    phi0      = TT.as_tensor_variable(phi0)
    derphi0   = TT.as_tensor_variable(derphi0)
    c1        = TT.as_tensor_variable(c1)
    c2        = TT.as_tensor_variable(c2)
    # Function reprensenting the computations of one step of the while loop
    def while_zoom(phi_rec, a_rec, a_lo, a_hi, phi_hi, phi_lo, derphi_lo,
                   *gfks):
        # interpolate to find a trial step length between a_lo and
        # a_hi Need to choose interpolation here.  Use cubic
        # interpolation and then if the result is within delta *
        # dalpha or outside of the interval bounded by a_lo or a_hi
        # then use quadratic interpolation, if the result is still too
        # close, then use bisection
        dalpha = a_hi-a_lo
        a = TT.switch( dalpha < zero, a_hi, a_lo)
        b = TT.switch( dalpha < zero, a_lo, a_hi)

        # minimizer of cubic interpolant
        # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
        #
        # if the result is too close to the end points (or out of the
        # interval) then use quadratic interpolation with phi_lo,
        # derphi_lo and phi_hi if the result is stil too close to the
        # end points (or out of the interval) then use bisection

        # cubic interpolation
        cchk = delta1*dalpha
        a_j_cubic = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec)

        # quadric interpolation
        qchk = delta2*dalpha
        a_j_quad = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
        cond_q = lazy_or('condq',TT.isnan(a_j_quad), a_j_quad > b-qchk, a_j_quad < a + qchk)
        a_j_quad = TT.switch(cond_q, a_lo +
                             numpy.asarray(0.5, dtype=theano.config.floatX)*dalpha, a_j_quad)


        # pick between the two ..
        cond_c = lazy_or('condc',TT.isnan(a_j_cubic), TT.bitwise_or(a_j_cubic > b -
                                                            cchk, a_j_cubic
                                                            < a + cchk))
        # this lazy if actually decides if we need to run the quadric
        # interpolation
        a_j = TT.switch(cond_c, a_j_quad, a_j_cubic)
        #a_j = ifelse(cond_c, a_j_quad,  a_j_cubic)

        # Check new value of a_j

        phi_aj = phi(a_j)
        derphi_aj, gfks_aj = derphi(a_j)

        stop = lazy_and('stop', TT.bitwise_and(phi_aj <= phi0 + c1*a_j*derphi0,
                         phi_aj < phi_lo),
                        abs(derphi_aj) <= -c2*derphi0)


        cond1 = TT.bitwise_or(phi_aj > phi0 + c1*a_j*derphi0,
                              phi_aj >= phi_lo)
        cond2 = derphi_aj*(a_hi - a_lo) >= zero

        # Switches just make more sense here because they have a C
        # implementation and they get composed
        phi_rec = ifelse( cond1, phi_hi,
                            TT.switch( cond2, phi_hi, phi_lo), name =
                         'phi_rec')
        a_rec   = ifelse( cond1, a_hi,
                            TT.switch( cond2, a_hi, a_lo), name='a_rec')
        a_hi    = ifelse( cond1, a_j,
                            TT.switch( cond2, a_lo, a_hi), name='a_hi')
        phi_hi  = ifelse( cond1, phi_aj,
                            TT.switch( cond2, phi_lo, phi_hi), name='phi_hi')

        a_lo      = TT.switch(cond1, a_lo, a_j)
        phi_lo    = TT.switch(cond1, phi_lo, phi_aj)
        derphi_lo = ifelse(cond1, derphi_lo, derphi_aj, name='derphi_lo')
        rgfks = [ifelse(cond1, x, x_aj) for x,x_aj in zip(gfks,
                                                                 gfks_aj)]

        a_star = a_j
        val_star = phi_aj
        valprime = ifelse(cond1, nan, TT.switch(cond2, derphi_aj,
                                                  nan), name='valprime')

        return ( [ phi_rec,
                  a_rec,
                  a_lo,
                  a_hi,
                  phi_hi,
                  phi_lo,
                  derphi_lo,
                  a_star,
                  val_star,
                  valprime] + rgfks,
                until(stop) )

    maxiter = 10
    delta1 = TT.constant(numpy.asarray(0.2,
                                       dtype=theano.config.floatX))  # cubic interpolant check
    delta2 = TT.constant(numpy.asarray(0.1,
                                       dtype=theano.config.floatX))  # quadratic interpolant check
    phi_rec = phi0
    a_rec = zero

    # Initial iteration

    dalpha = a_hi-a_lo
    a = TT.switch( dalpha < zero, a_hi, a_lo)
    b = TT.switch( dalpha < zero, a_lo, a_hi)
    #a = ifelse(dalpha < 0, a_hi, a_lo)
    #b = ifelse(dalpha < 0, a_lo, a_hi)

    # minimizer of cubic interpolant
    # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
    #
    # if the result is too close to the end points (or out of the
    # interval) then use quadratic interpolation with phi_lo,
    # derphi_lo and phi_hi if the result is stil too close to the
    # end points (or out of the interval) then use bisection


    # quadric interpolation
    qchk = delta2*dalpha
    a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
    cond_q = lazy_or('mcond_q',TT.isnan(a_j), TT.bitwise_or( a_j > b-qchk, a_j < a +
                                                  qchk))

    a_j = TT.switch(cond_q, a_lo +
                    numpy.asarray(0.5, dtype=theano.config.floatX)*dalpha, a_j)


    # Check new value of a_j

    phi_aj = phi(a_j)
    derphi_aj, gfks_aj = derphi(a_j)



    cond1 = TT.bitwise_or(phi_aj > phi0 + c1*a_j*derphi0,
                          phi_aj >= phi_lo)
    cond2 = derphi_aj*(a_hi - a_lo) >= zero

    # Switches just make more sense here because they have a C
    # implementation and they get composed
    phi_rec = ifelse( cond1, phi_hi,
                        TT.switch( cond2, phi_hi, phi_lo), name='mphirec')
    a_rec   = ifelse( cond1, a_hi,
                        TT.switch( cond2, a_hi, a_lo), name='marec')
    a_hi    = ifelse( cond1, a_j,
                        TT.switch( cond2, a_lo, a_hi), name='mahi')
    phi_hi  = ifelse( cond1, phi_aj,
                        TT.switch( cond2, phi_lo, phi_hi), name='mphihi')

    onlyif = lazy_and( 'only_if', TT.bitwise_and(phi_aj <= phi0 + c1*a_j*derphi0,
                       phi_aj < phi_lo),
                       abs(derphi_aj) <= -c2*derphi0)

    a_lo      = TT.switch(cond1, a_lo, a_j)
    phi_lo    = TT.switch(cond1, phi_lo, phi_aj)
    derphi_lo = ifelse(cond1, derphi_lo, derphi_aj, name = 'derphi_lo_main')
    sgfks = [ ifelse(cond1, x, x_aj) for x,x_aj in zip(gfks,
                                                              gfks_aj)]
    phi_rec.name = 'phi_rec'
    a_rec.name = 'a_rec'
    a_lo.name = 'a_lo'
    a_hi.name = 'a_hi'
    phi_hi.name = 'phi_hi'
    phi_lo.name = 'phi_lo'
    derphi_lo.name = 'derphi_lo'
    vderphi_aj = ifelse(cond1, nan, TT.switch(cond2, derphi_aj, nan),
                        name='vderphi_aj')
    outs, updates = theano.scan(while_zoom,
                                outputs_info= [phi_rec, a_rec, a_lo, a_hi,
                                               phi_hi, phi_lo, derphi_lo,
                                               None, None, None]# + sgfks,
                                + [None for
                                                                    k in
                                                                    sgfks],
                                non_sequences = sgfks,
                                n_steps = maxiter,
                                name = 'while_zoom',
                               mode = theano.Mode(linker='cvm_nogc'),
                               profile = profile)
    a_star   = ifelse(onlyif, a_j   , outs[7][-1], name='astar')
    val_star = ifelse(onlyif, phi_aj, outs[8][-1], name='valstar')
    valprime = ifelse(onlyif, vderphi_aj, outs[9][-1], name='valprime')
    valgfks  = [ ifelse(onlyif, vx, nwx[-1])
                for vx,nwx in zip(sgfks, outs[10:])]

    ## WARNING !! I ignore updates given by scan which I should not do !!!
    return a_star, val_star, valprime, valgfks



