import argparse	
from argparse import RawTextHelpFormatter
import numpy as np               
import matplotlib.pyplot as plt  
import integrators # Stepper functions and integrator driver

'''Define global variables for use in RHS of ODE'''
def set_odepar(par):
    global odepar
    odepar = par

'''Retrieve global variables for use in RHS of ODE'''
def get_odepar():
    global odepar
    return odepar

'''Calculates RHS for Lorenz System sender (Part 1)'''
def dydx_sender(t, values, dx):
    # Retrieve constants
    constants = get_odepar()
    sigma = constants[0]
    b = constants[1]
    r = constants[2]

    # Unpackage values input
    x = values[0]
    y = values[1]
    z = values[2]

    # Define RHS for system
    dydx = np.zeros(3)
    dydx[0] = sigma*(y-x)
    dydx[1] = r*x - y - x*z
    dydx[2] = x*y - b*z
    
    return dydx

''''''
def main():
    # Expect integrator and r value inputs
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("stepper", type=str, default='euler', help="Stepping function\n")
    parser.add_argument("r", type=float, default=10, help="Steepness of temperature gradient")
    args   = parser.parse_args()
    
    # Determine Stepper
    if (args.stepper == "euler"):
        fORD = integrators.euler
    elif (args.stepper == "rk2"):
        fORD = integrators.rk2
    elif (args.stepper == "rk4"):
        fORD = integrators.rk4
    elif (args.stepper == "rk45"):
        fORD = integrators.rk45
    elif (args.stepper == "backeuler"):
        fORD = integrators.backeuler
    else:
        raise Exception("invalid stepper %s" % (args.stepper))

    # Initialization
    # Assuming Lorenz system is always a initial-value problem
    nstep = 10000
    t0 = 0
    t1 = 100
    x0 = 10
    y0 = 10
    z0 = 10
    values0 = np.array([x0, y0, z0])

    # Needed constants
    sigma = 10
    b = 8.0/3
    r = args.r
    set_odepar(np.array([sigma, b, r]))

    # Set driver and RHS ODE definitions
    fINT = integrators.ode_ivp
    fRHS = dydx_sender

    # Solve for values for each of the dependent variables (X, Y, Z) and for independent variable (t)
    t, values, iterations = fINT(fRHS, fORD, t0, t1, values0, nstep)

    # Plot results
    plt.plot(t, values[0])

if __name__ == "__main__":
    main()