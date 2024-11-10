import argparse	
from argparse import RawTextHelpFormatter
import numpy as np               
import matplotlib.pyplot as plt  
from mpl_toolkits import mplot3d # Import toolkit for 3D plots
import integrators # Stepper functions and integrator driver
from systemparameters import SystemParameters
from scipy.fft import fft, fftfreq
import signals

'''
Calculates RHS for Lorenz System sender

Sender is responsible for calling the receiver in the event receiver values are specified
'''
def dydx_sender(t, values, dx):
    # Retrieve constants (class variables of SystemParameters)
    sigma = SystemParameters.sigma
    b = SystemParameters.b
    r = SystemParameters.r

    # Unpackage values  (constant for case of sender and receiver)
    x = values[0]
    y = values[1]
    z = values[2]

    # Define RHS for sender system
    dydx = np.zeros(3)
    dydx[0] = sigma*(y-x)
    dydx[1] = r*x - y - x*z
    dydx[2] = x*y - b*z
    
    # Check if receiver values are set in this time step
    if (len(values) > 3):
        receiver = dydx_receiver(t, values, dx)
        dydx = np.append(dydx, receiver, axis=0)
    
    return dydx

'''Calculates RHS for Lorenz System receiver'''
def dydx_receiver(t, values, dx):
    # Will need to come up with an intelligent way to check if the receiver is perturbed (just pass in another index with value of s? - say s= values[6] ?)
    
    # Retrieve constants (class variables of SystemParameters)
    sigma = SystemParameters.sigma
    b = SystemParameters.b
    r = SystemParameters.r

    # Unpackage values
    x = values[0] # Using x as the previous value and not the value calculated above
    u = values[3]
    v = values[4]
    w = values[5]

    # Define RHS for receiver system
    dydx = np.zeros(3)
    if SystemParameters.perturbation:
        # Add pertubration if present
        perturbation = SystemParameters.perturbation
        x_perturbed = x + perturbation[t-1]
        
        dydx[0] = sigma*(v-u)
        dydx[1] = r*x_perturbed - v - x_perturbed*w
        dydx[2] = x_perturbed*v - b*w
    else:
        dydx[0] = sigma*(v-u)
        dydx[1] = r*x - v - x*w
        dydx[2] = x*v - b*w
    
    return dydx

''''''
def main():
    # Expect integrator and r value inputs
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("stepper", type=str, default='euler', help="Stepping function")
    parser.add_argument("r", type=float, default=10, help="Steepness of temperature gradient")
    parser.add_argument("--receiver", action='store_true', help="Flag to include receiver ODEs in system") # Optional arguement for included additional ODEs
    parser.add_argument("--perturbation", type=str, default=None, help="Optional kind of perturbation to include in the system")  # Optional text argument
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
    
    # Initialization (assuming Lorenz system is always an initial-value problem)
    nstep = 10000
    t0 = 0
    t1 = 100
    
    x0 = 10
    y0 = 10
    z0 = 10
    values0 = [x0, y0, z0]
    
    # Determine presence of receiver system (in addition to sender)
    if args.receiver:
        u0 = 10
        v0 = 10
        w0 = 10
        values0.extend([u0, v0, w0])
    
    values0 = np.array(values0)
    
    # Determine presence of perturbation (on the receiver system)
    if (args.perturbation == "binary"):
        SystemParameters.perturbation = signals.square_wave_signal()
    elif (args.perturbation == "recording"):
        SystemParameters.perturbation = signals.load_audio_signal()

    # Needed constants (using standard values for the Lorenz System for sigma and and passing in steepness of temperature gradient r as arguement)
    sigma = 10
    b = 8.0/3
    
    SystemParameters.sigma = sigma
    SystemParameters.b = b
    SystemParameters.r = args.r

    # Set driver and RHS ODE definitions
    fINT = integrators.ode_ivp
    fRHS = dydx_sender

    # Solve for values for each of the dependent variables (X, Y, Z) and for independent variable (t)
    t, values, iterations = fINT(fRHS, fORD, t0, t1, values0, nstep)
    x = values[0]
    y = values[1]
    z = values[2]

    # FFT
    plt.plot(fftfreq(nstep, t[1]-t[0])[:nstep//2], 2.0/nstep * np.abs(fft(z)[0:nstep//2]))
    plt.xlabel('frequency')
    plt.ylabel('amplitude')
    plt.title('FFT for z(t)')
    plt.show()

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