import numpy as np
from systemparameters import SystemParameters

'''Forward Euler Step'''
def euler(fRHS,x0,y0,dx):
    y = y0 + dx*fRHS(x0, y0, dx)
    return y,1

'''Runge-Kutta 2nd-Order Step'''
def rk2(fRHS,x0,y0,dx):
    # Midpoint method
    k1 = fRHS(x0, y0, dx)
    y_mid = y0 + 0.5*dx*k1
    x_mid = x0 + 0.5*dx
    k2 = fRHS(x_mid, y_mid, dx)
    
    y = y0 + dx*k2
    return y,1

'''Runge-Kutta 4th-Order Step'''
def rk4(fRHS,x0,y0,dx):
    k1 = fRHS(x0, y0, dx)
    k2 = fRHS(x0 + 0.5*dx, y0 + 0.5*dx*k1, dx)
    k3 = fRHS(x0 + 0.5*dx, y0 + 0.5*dx*k2, dx)
    k4 = fRHS(x0 + dx, y0 + dx*k3, dx)
    
    y = y0 + (dx/6) * (k1 + 2*k2 + 2*k3 + k4)
    return y,1

'''Runge-Kutta 4/5 Single Step'''
def rk45single(fRHS,x0,y0,dx):
    a = np.array([0.0,0.2,0.3,0.6,1.0,0.875]) # weights for x
    b = np.array([[0.0           , 0.0        , 0.0          , 0.0             , 0.0         ],
                  [0.2           , 0.0        , 0.0          , 0.0             , 0.0         ],
                  [0.075         , 0.225      , 0.0          , 0.0             , 0.0         ],
                  [0.3           , -0.9       , 1.2          , 0.0             , 0.0         ],
                  [-11.0/54.0    , 2.5        , -70.0/27.0   , 35.0/27.0       , 0.0         ],
                  [1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0]])
    
    c = np.array([37.0/378.0,0.0,250.0/621.0,125.0/594.0,0.0,512.0/1771.0])
    dc = np.array([2825.0/27648.0,0.0,18575.0/48384.0,13525.0/55296.0,277.0/14336.0,0.25])
    dc = c-dc
    n = y0.size
    dy = np.zeros(n)        # updates (arguments in f(x,y))
    dydx = np.zeros((6,n))    # derivatives (k1,k2,k3,k4,k5,k6)
    yout = y0                 # result
    yerr = np.zeros(n)        # error
    dydx[0,:] = dx*fRHS(x0,y0,dx)  # first guess
    for i in range(1,6):           # outer loop over k_i 
        dy[:]     = 0.0
        for j in range(i):         # inner loop over y as argument to fRHS(x,y)
            dy = dy + b[i,j]*dydx[j,:]
        dydx[i,:] = dx*fRHS(x0+a[i]*dx,y0+dy,a[i]*dx)
    for i in range(0,6):           # add up the k_i times their weighting factors
        yout = yout + c [i]*dydx[i,:]
        yerr = yerr + dc[i]*dydx[i,:]

    return yout,yerr

'''Runge-Kutta 4/5 Adaptive Step'''
def rk45(fRHS,x0,y0,dx):
    pshrink = -0.25
    pgrow   = -0.2
    safe    = 0.9
    errcon  = 1.89e-4
    eps     = 1e-6                                          # this should depend on problem. Accuracy.
    maxit   = 100000                                        # this should depend on problem
    xt      = 0.0                                           # temporary independent variable: will count up to dx.
    x1      = x0
    it      = 0                                             # iteration counter, as safeguard
    n       = y0.size
    dydx    = fRHS(x0,y0,dx)
    y1      = y0
    y2      = y0
    dxtry   = dx                                            # starting guess for step size    
    dxtmp   = dx
    idone   = 0
    while ((xt < dx) and (it < maxit)):
        yscal = np.abs(y1) + np.abs(dxtry*dydx)             # error scaling on last timestep, see NR92, sec 16.2
        idone = 0                                           # reset idone
        while (not idone):                                  # figure out an acceptable stepsize
            y2,yerr = rk45single(fRHS,x1,y1,dxtry)     # keep recalculating y2 using current trial step
            errmax  = np.max(np.abs(yerr/yscal)/eps)
            if (errmax > 1.0):                              # stepsize too large - reduce
                dxtmp = dxtry*safe*np.power(errmax,pshrink)
                dxtry = np.max(np.array([dxtmp,0.1*dxtry])) # warning! This is only for dxtry > 0.
                xnew  = xt + dxtry
                if (xnew == xt):
                    raise Exception('[step_rk45]: xnew == xt. dx1 = %13.5e' % (dxtry))
            else:                                           # stepsize ok - we're done with the trial loop
                idone = 1
        y1 = y2                                             # update so that integration is advanced at next iteration.
        it = it+1
        if (errmax > errcon):                               # if the error is larger than safety, reduce growth rate
            dxnext = safe*dxtry*np.power(errmax,pgrow)
        else:                                               # if error less than safety, increase by factor of 5.
            dxnext = 5.0*dxtry
        x1    = x1 + dxtry
        xt    = xt + dxtry
        dxtry = np.min(np.array([dx-xt,dxnext]))            # guess next timestep - make sure it's flush with dx.
    return y2,it

'''Backward Euler Step'''
def backeuler(fRHS,x0,y0,dx):
    # Retrieve system parameters
    sigma = SystemParameters.sigma
    b = SystemParameters.b
    r = SystemParameters.r
    
    # Pad y0 with zeros if the input vector has less than 6 variables
    # y0 = np.concatenate([y0, np.zeros(6 - len(y0))]) if len(y0) < 6 else y0
    
    # Assign current variables values (doing this as the previous step and not after solving for the next step)
    x = y0[0]
    y = y0[1]
    z = y0[2]
    #u = y0[3]
    #v = y0[4]
    #w = y0[5]
    
    # Hard code 3x3 Jacobian matrix
    jacobian3 = np.array([[-sigma, sigma, 0], 
                         [(r-z), -1, -x], 
                         [y, x, -b]]) 
    
    # Pad y0 with zeros if the input vector has less than 6 variables
    # y0 = np.concatenate([y0, np.zeros(6 - len(y0))]) if len(y0) < 6 else y0
    #
    # Hard code 6x6 Jacobian matrix
    # jacobian6 = np.array([[-sigma, sigma, 0, 0, 0, 0], 
    #                     [(r-z), -1, -x, 0, 0, 0], 
    #                     [y, x, -b, 0, 0, 0], 
    #                     [0, 0, 0, -sigma, sigma, 0], 
    #                     [(r-w), 0, 0, 0, -1, x], 
    #                     [v, 0, 0, 0, x, -b]]) 
    
    rhs_evaluation = fRHS(x0, y0, dx)
                         
    y = y0 + dx* np.dot(np.linalg.inv(np.identity(3) - dx*jacobian3), rhs_evaluation)
    return y,1

'''Driver for initial value problem'''
def ode_ivp(fRHS, fORD, t0, t1, values0, nstep):
    nvar = values0.size # Number of ODEs
    independent = np.linspace(t0,t1,nstep+1) # Generates equal-distant support points
    dependents = np.zeros((nvar,nstep+1)) # Result array 
    dependents[:,0] = values0 # Set initial conditions
    dx = independent[1] - independent[0] # Step size
    it = np.zeros(nstep+1)

    for k in range(1,nstep+1):
        dependents[:,k], it[k] = fORD(fRHS,independent[k-1], dependents[:,k-1], dx)

    return independent, dependents, it