#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:09:16 2018
This code is an attempt to simulate my current coil configuration.
@author: ccartagena
"""

## Example use of code
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import the two functions from the magnetics package
# magnetics.py must be in the same directory as this file
# or in your PYTHONPATH environment variable
from magnetics import compute_greens, compute_greens_mp

# Define useful function to plot fields
def plot_fields(R,Z,psi,BR,BZ):
    # Plot the fields!
    """
    
    The line below is attaching four objects into four variables: fig, ax1, ax2
    ax3. The fig variable contains the entire object, while the other three 
    contain the three subplots of the huge object, fig.
    
    """
    fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(16,9))
    
    """
    
    The next two lines is creating a instance of the axes of subplot ax1 and 
    editing the position.
    
    """
    div1 = make_axes_locatable(ax1)
    cax1 = div1.append_axes("right","5%",pad="3%")
    
    """
    
    The next set of lines is plotting the figure, I do not know why the locator 
    is included since it is not referenced anywhere else.
    
    What is interesting is the line where the color bar is created,
    
    cbar1 = fig.colorbar(cf1, cax = cax1) -> this creates a colorbar in the 
    figure, separate from the subplot. It is placed in the location of the 
    corresponding subplot by the cax = cax1. Nice
    
    """

    ax1.contour(R,Z,psi,50,colors="k")
    cf1 = ax1.contourf(R,Z,np.sqrt(BR**2+BZ**2),locator=ticker.LogLocator())
    cbar1 = fig.colorbar(cf1,cax=cax1)
    cbar1.set_label("|B| (T)")
    ax1.set_aspect(1)
    ax1.set_xlabel("R (m)")
    ax1.set_ylabel("Z (m)")
    # The two lines below need to be changed when the radial and axial 
    # normalizations change.
    ax1.plot([0.247,0.247],[-1,1],'-',color = 'white',linewidth = 2)
    ax1.plot([0.617,0.617],[-1,1],'-',color = 'white',linewidth = 2)
    """
    The next three sets of commands follow the same logic as the first.
    """
    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes("right","5%",pad="3%")
    
    ax2.contour(R,Z,psi,50,colors="k")
    cf2 = ax2.contourf(R,Z,np.abs(BR),locator=ticker.LogLocator())
    cbar2 = fig.colorbar(cf2,cax=cax2)
    cbar2.set_label("|B_R| (T)") # I need to change the white space.
    ax2.set_aspect(1)
    ax2.set_xlabel("R (m)")
 #   ax2.set_ylabel("Z (m)")
    # The two lines below need to be changed when the radial and axial 
    # normalizations change.
    ax2.plot([0.247,0.247],[-1,1],'-',color = 'white',linewidth = 2)
    ax2.plot([0.617,0.617],[-1,1],'-',color = 'white',linewidth = 2)
    
    div3 = make_axes_locatable(ax3)
    cax3 = div3.append_axes("right","5%",pad="3%")
    
    cf3 = ax3.contourf(R,Z,np.abs(BZ),locator=ticker.LogLocator())
    ax3.contour(R,Z,psi,50,colors="k")
    cbar3 = fig.colorbar(cf3,cax=cax3)
    cbar3.set_label("|B_Z| (T)")
    ax3.set_aspect(1)
    ax3.set_xlabel("R (m)")
#    ax3.set_ylabel("Z (m)")
    # The two lines below need to be changed when the radial and axial 
    # normalizations change.
    ax3.plot([0.247,0.247],[-1,1],'-',color = 'white',linewidth = 2)
    ax3.plot([0.617,0.617],[-1,1],'-',color = 'white',linewidth = 2)
    
    plt.tight_layout()
    plt.savefig('OrientationIIv4.pdf')
    plt.show()

# This section will show you how to use the code through a 
# couple specific examples.

# Build mxn R-Z grid (region of space where you want to 
# calculate the magnetic flux and field)
m = 1001
n = 2001
r = np.linspace(0,1,m)
z = np.linspace(-1,1,n)
R,Z = np.meshgrid(r,z)

#################################################################
"""
# Loop 1: single current loop at R = 1.0 m Z = 1.0 m
#
rlocs1 = [0.95]
zlocs1 = [0.25]
cur_dir1 = [-1]
# zip current locations and directions into a list of 3-tuples (needed
# for compute_greens function)
rzdir1 = zip(rlocs1,zlocs1,cur_dir1)
# create greens functions for magnetic flux, and 2 components of B
# these matrices are m*n x n_coils
gpsi1, gBR1, gBZ1 = compute_greens(R.flatten(),Z.flatten(),rzdir1)
# specify number of amps in each of the coils, here we are using 100 amps
# in 1 coil
coil_currents1 = 100*np.ones(1)
# take dot product of greens function for desired field and coil currents
# and reshape to same shape as the 2D RZ grid
psi1 = (gpsi1.dot(coil_currents1)).reshape(R.shape)
BR1 = (gBR1.dot(coil_currents1)).reshape(R.shape)
BZ1 = (gBZ1.dot(coil_currents1)).reshape(R.shape)
"""

# Build solenoid 1 (outer solenoid)
# h number of turns in z direction
# w number of turns in r direction
h1 = 20
w1 = 5
n_coils1 = h1*w1
rlocs1 = np.linspace(0.948,1,w1)
zlocs1 = np.linspace(-0.025,0.228,h1)
rlocs1,zlocs1 = np.meshgrid(rlocs1,zlocs1)
rlocs1 = rlocs1.flatten()
zlocs1 = zlocs1.flatten()
cur_dir1 = np.ones(n_coils1)
# zip current locations and directions into a list of 3-tuples (needed
# for compute_greens function)
rzdir1 = zip(rlocs1,zlocs1,cur_dir1)
gpsi1, gBR1, gBZ1 = compute_greens(R.flatten(),Z.flatten(),rzdir1)

coil_currents1 = +180*np.ones(n_coils1)

psi1 = (gpsi1.dot(coil_currents1)).reshape(R.shape)
BR1 = (gBR1.dot(coil_currents1)).reshape(R.shape)
BZ1 = (gBZ1.dot(coil_currents1)).reshape(R.shape)


################################
"""
# Second current loop

rlocs2 = [0.95]
zlocs2 = [-0.25]
cur_dir2 = [-1]
# zip current locations and directions into a list of 3-tuples (needed
# for compute_greens function)
rzdir2 = zip(rlocs2,zlocs2,cur_dir2)
# create greens functions for magnetic flux, and 2 components of B
# these matrices are m*n x n_coils
gpsi2, gBR2, gBZ2 = compute_greens(R.flatten(),Z.flatten(),rzdir2)
# specify number of amps in each of the coils, here we are using 100 amps
# in 1 coil
coil_currents2 = 100*np.ones(1)
# take dot product of greens function for desired field and coil currents
# and reshape to same shape as the 2D RZ grid
psi2 = (gpsi2.dot(coil_currents2)).reshape(R.shape)
BR2 = (gBR2.dot(coil_currents2)).reshape(R.shape)
BZ2 = (gBZ2.dot(coil_currents2)).reshape(R.shape)

"""
# Build solenoid 2 ( outer solenoid)
# h number of turns in z direction
# w number of turns in r direction
h2 = 20
w2 = 5
n_coils2 = h2*w2
rlocs2 = np.linspace(0.948,1,w2)
zlocs2 = np.linspace(0.506,0.759,h2)
rlocs2,zlocs2 = np.meshgrid(rlocs2,zlocs2)
rlocs2 = rlocs2.flatten()
zlocs2 = zlocs2.flatten()
cur_dir2 = np.ones(n_coils2)
# zip current locations and directions into a list of 3-tuples (needed
# for compute_greens function)
rzdir2 = zip(rlocs2,zlocs2,cur_dir2)
gpsi2, gBR2, gBZ2 = compute_greens(R.flatten(),Z.flatten(),rzdir2)

coil_currents2 = -180*np.ones(n_coils2)

psi2 = (gpsi2.dot(coil_currents2)).reshape(R.shape)
BR2 = (gBR2.dot(coil_currents2)).reshape(R.shape)
BZ2 = (gBZ2.dot(coil_currents2)).reshape(R.shape)
#########################
"""
# Third loop


rlocs3 = [0.25]
zlocs3 = [0.0]
cur_dir3 = [-1]
# zip current locations and directions into a list of 3-tuples (needed
# for compute_greens function)
rzdir3 = zip(rlocs3,zlocs3,cur_dir3)
# create greens functions for magnetic flux, and 2 components of B
# these matrices are m*n x n_coils
gpsi3, gBR3, gBZ3 = compute_greens(R.flatten(),Z.flatten(),rzdir3)
# specify number of amps in each of the coils, here we are using 100 amps
# in 1 coil
coil_currents3 = 1500*np.ones(1)
# take dot product of greens function for desired field and coil currents
# and reshape to same shape as the 2D RZ grid
psi3 = (gpsi3.dot(coil_currents3)).reshape(R.shape)
BR3 = (gBR3.dot(coil_currents3)).reshape(R.shape)
BZ3 = (gBZ3.dot(coil_currents3)).reshape(R.shape)
"""
# Build solenoid 3 ( inner solenoid)
# h number of turns in z direction
# w number of turns in r direction
h3 = 25
w3 = 4
n_coils3 = h3*w3
rlocs3 = np.linspace(0.152,.167,w3)
zlocs3 = np.linspace(0.038,0.578,h3)
rlocs3,zlocs3 = np.meshgrid(rlocs3,zlocs3)
rlocs3 = rlocs3.flatten()
zlocs3 = zlocs3.flatten()
cur_dir3 = np.ones(n_coils3)
# zip current locations and directions into a list of 3-tuples (needed
# for compute_greens function)
rzdir3 = zip(rlocs3,zlocs3,cur_dir3)
gpsi3, gBR3, gBZ3 = compute_greens(R.flatten(),Z.flatten(),rzdir3)

coil_currents3 = +2800*np.ones(n_coils3)

psi3 = (gpsi3.dot(coil_currents3)).reshape(R.shape)
BR3 = (gBR3.dot(coil_currents3)).reshape(R.shape)
BZ3 = (gBZ3.dot(coil_currents3)).reshape(R.shape)








BR = BR1 + BR2 + BR3
BZ = BZ1 + BZ2 + BZ3
psi = psi1 + psi2 + psi3


# Note: All quantities are in SI units
# plot_fields(R,Z,psi,BR,BZ)
#################################################################
"""
#################################################################
# Example 3: Two concentric solenoids
#
# Build solenoid 1 ( inner solenoid)
# h number of turns in z direction
# w number of turns in r direction
h1 = 25
w1 = 4
n_coils1 = h1*w1
rlocs1 = np.linspace(0,.5,w1)
zlocs1 = np.linspace(-.1,.1,h1)
rlocs1,zlocs1 = np.meshgrid(rlocs1,zlocs1)
rlocs1 = rlocs1.flatten()
zlocs1 = zlocs1.flatten()
cur_dir1 = np.ones(n_coils1)
# zip current locations and directions into a list of 3-tuples (needed
# for compute_greens function)
rzdir1 = zip(rlocs1,zlocs1,cur_dir1)
gpsi1, gBR1, gBZ1 = compute_greens(R.flatten(),Z.flatten(),rzdir1)

# Build solenoid 2 ( outer solenoid)
# h number of turns in z direction
# w number of turns in r direction
h2 = 50
w2 = 2
n_coils2 = h2*w2
rlocs2 = np.linspace(0.95,1,w1)
zlocs2 = np.linspace(-.1,.1,h1)
rlocs2,zlocs2 = np.meshgrid(rlocs2,zlocs2)
rlocs2 = rlocs2.flatten()
zlocs2 = zlocs2.flatten()
cur_dir2 = np.ones(n_coils2)
# zip current locations and directions into a list of 3-tuples (needed
# for compute_greens function)
rzdir2 = zip(rlocs2,zlocs2,cur_dir2)
gpsi2, gBR2, gBZ2 = compute_greens(R.flatten(),Z.flatten(),rzdir2)

# Now we have two sets of greens functions for flux and magnetic field
# components: one for each solenoid. You can take dot products of each of 
# these greens functions with the currents you want in each coil and add
# the results together to get the total field.
#
# For example, let's put 150 A through the inner solenoid and -200 A through
# the outer one and plot the result. Then, change the outer solenoid current
# to 300 A and plot again.
coil_currents1 = 150*np.ones(n_coils1)
coil_currents2 = -200*np.ones(n_coils2)

psi1 = (gpsi1.dot(coil_currents1)).reshape(R.shape)
BR1 = (gBR1.dot(coil_currents1)).reshape(R.shape)
BZ1 = (gBZ1.dot(coil_currents1)).reshape(R.shape)
psi2 = (gpsi2.dot(coil_currents2)).reshape(R.shape)
BR2 = (gBR2.dot(coil_currents2)).reshape(R.shape)
BZ2 = (gBZ2.dot(coil_currents2)).reshape(R.shape)
psi = psi1 + psi2
BR = BR1 + BR2
BZ = BZ1 + BZ2
plot_fields(R,Z,psi,BR,BZ)

#Now change current in solenoid 2 and plot again
coil_currents2 = -300*np.ones(n_coils2)
psi2 = (gpsi2.dot(coil_currents2)).reshape(R.shape)
BR2 = (gBR2.dot(coil_currents2)).reshape(R.shape)
BZ2 = (gBZ2.dot(coil_currents2)).reshape(R.shape)
psi = psi1 + psi2
BR = BR1 + BR2
BZ = BZ1 + BZ2
plot_fields(R,Z,psi,BR,BZ)
"""