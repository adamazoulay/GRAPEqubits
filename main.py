'''
This is the second version for the implementation and
testing of the GRAPE algorithm from here:
http://www.org.chemie.tu-muenchen.de/glaser/94(GRAPE_JMR_05).pdf

In this version we impliment the control for large driving forces
in a rotating frame for a single qubit using fourier analysis to
limit the spectrum of frequencies we use for the driving force.
This also means we have to use a custom performance function.

Adam Azoulay
March 25th, 2015
v0.2
'''

from qutip import * #for unitary operators and testing
import numpy as np
from scipy import linalg as la
from math import pi, sin, cos, exp, floor
import cmath
import random as rand
import matplotlib.pyplot as plt
import sys

#This function if for initialization of our control vectors
#In this case, these are just the Fourier coefficients
def genCVec():

    vec = np.zeros((4,n_max))

    max = 100   
    for i in range(n_max):
        vec[0][i] = 0 #rand.randrange(1, max)  #b_In
        vec[1][i] = rand.randrange(1, max)  #c_In
        vec[2][i] = 0 #rand.randrange(1, max)  #b_Qn
        vec[3][i] = rand.randrange(1, max)  #c_Qn
    
    return vec


#This function takes in a matrix and will output the matrix exponential
#  of the form U = exp(-i*dt*A), where a is the matrix in an array
def getUnitary(A):
    
    Udata = la.expm(-1j*dt*A)
    U = Qobj(Udata)
    return U

#This function builds the waves for each time slice and places the
# results into a vector for use in the current iteration
def getCoefs():
    #cVec here contains [[b_In], [c_In], [b_Qn], [c_Qn]]
    for j in range(N):
        sumI = 0
        sumQ = 0
        t = j*dt
        for n in range(n_max):
            w_n = 2*pi*n/T

            sumI += cVec[0][n]*cos(w_n*t) + cVec[1][n]*sin(w_n*t)
            sumQ += cVec[2][n]*cos(w_n*t) + cVec[3][n]*sin(w_n*t)

        waves[j][0] = cos(w01*t)*(cos(w_d*t)*sumI + sin(w_d*t)*sumQ)
        waves[j][1] = sin(w01*t)*(cos(w_d*t)*sumI + sin(w_d*t)*sumQ)
        

    

#This function calculates the preformance function value Phi_0
def performFn(cVec):
    state = U0
    getCoefs()                              #Build the wave vectors
    for j in range(N):
        #Here is where it gets tricky. The control vector contains values for
        # b_In, c_In, b_Qn, and c_Qn. This means here we need construct the
        # coefficients for our HVec from the cVec and the spectrum restrictions.
        # This function returns u_x(t) and u_y(t)
        Htot = waves[j][0] * HVec[0] +\
               waves[j][1] * HVec[1]

        U[j] = getUnitary(Htot.full())
        state = U[j]*state

    #Now the state is fully evolved (ie state = X_j), we find <Pj|Xj><Xj|Pj>
    result1 = (target.dag() * state)
    result2 = (state.dag() * target)
    performance = (result1.tr() * result2.tr()).real/normalization
    return performance                       


#This function will update the X_j vector for all j given the HVec and cVec
def xpUpdate(cVec):
    Xj = U0
    Pj = target
    for j in range(1, N+1):
        Xj = U[j-1]*Xj                                 
        XVec[j-1] = Xj

    for j in range(N, 0, -1):
        Pj = U[j-1].dag()*Pj
        PVec[j-1] = Pj

    return XVec, PVec

#This function will update the gradient vector using X, P, and H
def dcUpdate():
    dcVec = np.zeros((4,n_max))
    for n in range(n_max):
        
        grad_1 = 0 #dphi/d(b_In)
        grad_2 = 0 #dphi/d(c_In)
        grad_3 = 0 #dphi/d(b_Qn)
        grad_4 = 0 #dphi/d(c_Qn)
        for j in range(N):
            t = j*dt
            w_n = 2*pi*n/T

            dpdu_1 = -1*((PVec[j].dag()*1j*dt*HVec[0]*XVec[j]).tr()*\
                               (XVec[j].dag()*PVec[j]).tr()).real

            dpdu_2 = -1*((PVec[j].dag()*1j*dt*HVec[1]*XVec[j]).tr()*\
                               (XVec[j].dag()*PVec[j]).tr()).real
            
            grad_1 += cos(w_d*t)*cos(w_n*t)*(dpdu_1*cos(w01*t) + dpdu_2*sin(w01*t))
            grad_2 += cos(w_d*t)*sin(w_n*t)*(dpdu_1*cos(w01*t) + dpdu_2*sin(w01*t))
            grad_3 += sin(w_d*t)*cos(w_n*t)*(dpdu_1*cos(w01*t) + dpdu_2*sin(w01*t))
            grad_4 += sin(w_d*t)*sin(w_n*t)*(dpdu_1*cos(w01*t) + dpdu_2*sin(w01*t))

        dcVec[0][n] = 0 #grad_1
        dcVec[1][n] = grad_2
        dcVec[2][n] = 0 #grad_3
        dcVec[3][n] = grad_4
                                                   
    return dcVec

#Just updates the gradient vector
def applydc():
    e = 5e6
    for j in range(4):
        for x in range(n_max):
            cVec[j][x] += e*dcVec[j][x]

    return cVec

#=======================================================================================
#   Plotting functions
#=======================================================================================
#I(j) and Q(j) functions here
def waves_plot(j):
    sumI = 0
    sumQ = 0
    t = dt*j
    for n in range(n_max):
        w_n = 2*pi*n/T        
        sumI += cVec[0][n]*cos(w_n*t) + cVec[1][n]*sin(w_n*t)
        sumQ += cVec[2][n]*cos(w_n*t) + cVec[3][n]*sin(w_n*t)

    return sumI, sumQ


#Plot results
def plotResults(num):
    plt.clf()   #Clear the plot
    x1 = np.arange(0, (N+1)*dt, dt)
    x2 = x1

    y1 = [0]
    y2 = [0]

    for j in range(N):
        Ij, Qj = waves_plot(j)
        y1.append(Ij)
        y2.append(Qj)

    
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.axhline(0, color='black')
    #plt.axis((0,T,-2e3,2e3))
    perform = "%0.2f" % phi
    plt.title('Pulse shapes for I(t) and Q(t)\nPhi= ' + perform)
    plt.ylabel('I(t)')

    plt.subplot(2, 1, 2)
    plt.plot(x2, y2)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.axhline(0, color='black')
    #plt.axis((0,T,-2e3,2e3))
    plt.xlabel('time (s)')
    plt.ylabel('Q(t)')

    plt.savefig('test/Rotation' + str(num) + '.png')
#=======================================================================================

    
    
#start our main function here
if __name__ == '__main__':

    #Initialize constants
    #!!!These are all globals and are used as such throughout the code!!!
    N = 475                                         #Total number of points for each control vector pulse
    w01 = 2*pi*5e4 #Hz
    T = 100*2*pi/w01 #seconds
    w_d = w01                                       #Driving frequency
    dw = 5*2*pi/T #Hz                               #So here we set our delta omega, i.e. or our
                                                    # allowed spectrum values (from n=0 -> n=n_max)
    H0 = 0                                          #We are in the RRF so H0 = 0
    
    dt = T/N #seconds/step
    n_max = int(floor(dw/(2*pi/T))) + 1             #Find our max n value
    U0 = Qobj([[1,0], [0,1]])                       #Initial U is the identity matrix

    cVec = genCVec()                                #Initialize control vectors
    dcVec = np.zeros((4,n_max))                     #Initialize the dphi/du_k vector
    
    XVec = np.empty(N, dtype=object)                #Create our XVec (for Xj=(U_j...U_1))
    PVec = np.empty(N, dtype=object)                #Create our PVec (for Pj=(U.dag_j+1...U.dag_N*U_F))
    HVec = np.empty(2, dtype=object)                #Create our H_k vector
    HVec = [sigmax(), sigmay()]                     #For this case our HVec is just sigmax and sigmay
    waves = np.zeros((N, 2))                        #This is where we will store the waves at all time
                                                    # slices j
    U = np.empty(N, dtype=object)                   #This is the vector of unitary evolution operators
                                                    # for all time slices j

    #This is used to set our target operator
    Hadamard = Qobj([[0.707,0.707],[0.707,-0.707]])
    Tgate = Qobj([[1,0],[0,cmath.exp(1j*pi/4)]])    #T phase gate
    Rx = Qobj([[0.707,-0.707j],[-0.707j,0.707]])    #pi around x axis
    target = Hadamard

    normalization = 4                               #Just the normalization for the performance function
    
    phi = performFn(cVec)
    print 'Phi: ', phi
    test = 0
    #Now that we are set up, we run the main loop to calculate the pulse shapes
    #  according to the GRAPE algorithm
    while 1:

        #calculate our X_j and P_j
        XVec, PVec = xpUpdate(cVec)
        dcVec = dcUpdate()
        cVec = applydc()        

        #find our new phi value
        phi = performFn(cVec)
        plotResults(test)
        test += 1
        print 'Phi: ', phi

    
        





