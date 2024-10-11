import argparse	
from argparse import RawTextHelpFormatter
import numpy as np               
import matplotlib.pyplot as plt  
from mpl_toolkits import mplot3d # Import toolkit for 3D plots
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
    x = values[0]
    y = values[1]
    z = values[2]

    # Plots for all values against t
    plt.figure(num=1,figsize=(5,6),dpi=100,facecolor='white')
    
    # X vs. T
    plt.subplot(311)
    plt.plot(t, x, linestyle='-', color='black', linewidth=1.0)
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.title('x vs. t')
    
    # Y vs. T
    plt.subplot(312)
    plt.plot(t, y, linestyle='-', color='black', linewidth=1.0)
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title('y vs. t')
    
    # Z vs. T
    plt.subplot(313)
    plt.plot(t, z, linestyle='-', color='black', linewidth=1.0)
    plt.xlabel('t')
    plt.ylabel('z(t)')
    plt.title('z vs. t')

    plt.tight_layout()
    plt.show()
    
    # Plot for all values against t (same plot)
    plt.figure(num=2,figsize=(5,6),dpi=100,facecolor='white')

    plt.plot(t, x, linestyle='-', linewidth=1.0, label = 'x')
    plt.plot(t, y, linestyle='-', linewidth=1.0, label = 'y')
    plt.plot(t, z, linestyle='-', linewidth=1.0, label = 'z')
    plt.xlabel('t')
    plt.ylabel('Functions of t')
    plt.title('x, y, z vs. t')
    plt.legend()
    plt.show()
    
    # Plots for all pairwise plots
    plt.figure(num=3,figsize=(5,6),dpi=100,facecolor='white')
    
    plt.subplot(311)
    plt.plot(x, y, linestyle='-', color='black', linewidth=1.0)
    plt.xlabel('x(t)')
    plt.ylabel('y(t)')
    plt.title('y vs. x')
    
    plt.subplot(312)
    plt.plot(x, z, linestyle='-', color='black', linewidth=1.0)
    plt.xlabel('x(t)')
    plt.ylabel('z(t)')
    plt.title('z vs. x')
    
    plt.subplot(313)
    plt.plot(y, z, linestyle='-', color='black', linewidth=1.0)
    plt.xlabel('y(t)')
    plt.ylabel('z(t)')
    plt.title('z vs. y')
    
    plt.tight_layout()
    plt.show()
    
    # Plot for 3D graph of all variables
    figure = plt.figure(num=3,figsize=(5,6),dpi=100,facecolor='white')
    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, color='black')
    ax.set_xlabel('x(t)')
    ax.set_ylabel('y(t)')
    ax.set_zlabel('z(t)')
    ax.set_title('3D plot of x(t), y(t), z(t)')
    plt.show()

if __name__ == "__main__":
    main()