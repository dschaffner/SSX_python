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
    
    ax1.contour(R,Z,psi,30,colors="k")
    cf1 = ax1.contourf(R,Z,np.sqrt(BR**2+BZ**2),101,locator=ticker.LogLocator())
    cbar1 = fig.colorbar(cf1,cax=cax1)
    cbar1.set_label("|B| (T)")
    ax1.set_aspect(1)
    ax1.set_xlabel("R (m)")
    ax1.set_ylabel("Z (m)")
    
    """
    The next three sets of commands follow the same logic as the first.
    """
    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes("right","5%",pad="3%")
    
    cf2 = ax2.contourf(R,Z,BR,30)
    cbar2 = fig.colorbar(cf2,cax=cax2)
    cbar2.set_label("B_R (T)")
    ax2.contour(R,Z,psi,30,colors="k")
    ax2.set_aspect(1)
    ax2.set_xlabel("R (m)")
    ax2.set_ylabel("Z (m)")
    
    div3 = make_axes_locatable(ax3)
    cax3 = div3.append_axes("right","5%",pad="3%")
    
    cf3 = ax3.contourf(R,Z,BZ,30)
    ax3.contour(R,Z,psi,30,colors="k")
    cbar3 = fig.colorbar(cf3,cax=cax3)
    cbar3.set_label("B_Z (T)")
    ax3.set_aspect(1)
    ax3.set_xlabel("R (m)")
    ax3.set_ylabel("Z (m)")
    
    plt.tight_layout()
    plt.show()

# This section will show you how to use the code through a 
# couple specific examples.

# Build mxn R-Z grid (region of space where you want to 
# calculate the magnetic flux and field)
m = 101
n = 201
r = np.linspace(0,1,m)
z = np.linspace(-1,1,n)
R,Z = np.meshgrid(r,z)
"""
#################################################################
# Example 1: single current loop at R = 0.5 m Z = 0.0 m
#
rlocs = [0.5]
zlocs = [0.0]
cur_dir = [1]
# zip current locations and directions into a list of 3-tuples (needed
# for compute_greens function)
rzdir = zip(rlocs,zlocs,cur_dir)
# create greens functions for magnetic flux, and 2 components of B
# these matrices are m*n x n_coils
gpsi, gBR, gBZ = compute_greens(R.flatten(),Z.flatten(),rzdir)
# specify number of amps in each of the coils, here we are using 100 amps
# in 1 coil
coil_currents = 100*np.ones(1)
# take dot product of greens function for desired field and coil currents
# and reshape to same shape as the 2D RZ grid
psi = (gpsi.dot(coil_currents)).reshape(R.shape)
BR = (gBR.dot(coil_currents)).reshape(R.shape)
BZ = (gBZ.dot(coil_currents)).reshape(R.shape)
# Note: All quantities are in SI units
plot_fields(R,Z,psi,BR,BZ)
#################################################################
"""
"""
#################################################################
# Example 2: Solenoid
#
# specify the current locations as coordinate pairs r,z in meters:
# 10 currents located at a radius of .35 m and linearly spaced from
# z = -.5m to +.5. cur_dir specifies the direction of each current 
# with +1 equivalent to the Z cross R direction and negative is R cross Z
n_coils = 10
rlocs = .35*np.ones(n_coils) 
zlocs = np.linspace(-.5,.5,n_coils)
cur_dir = np.ones(n_coils)
# zip current locations and directions into a list of 3-tuples (needed
# for compute_greens function)
rzdir = zip(rlocs,zlocs,cur_dir)
# create greens functions for magnetic flux, and 2 components of B
# these matrices are m*n x n_coils
gpsi, gBR, gBZ = compute_greens(R.flatten(),Z.flatten(),rzdir)
# specify number of amps in each of the coils, here we are using 150 amps
# in each turn
coil_currents = 150*np.ones(n_coils)
# take dot product of greens function for desired field and coil currents
# and reshape to same shape as the 2D RZ grid
psi = (gpsi.dot(coil_currents)).reshape(R.shape)
BR = (gBR.dot(coil_currents)).reshape(R.shape)
BZ = (gBZ.dot(coil_currents)).reshape(R.shape)
plot_fields(R,Z,psi,BR,BZ)
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
#################################################################
