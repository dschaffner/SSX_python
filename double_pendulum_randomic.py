import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Pendulum rod lengths (m), bob masses (kg).
L1, L2 = 0.1, 0.1
m1, m2 = 1, 1
# The gravitational acceleration (m.s-2).
g = 9.81
g = 4.9
g = 20.0
gs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
gs = [9.81]

def deriv(y, t, L1, L2, m1, m2, g):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, z1, theta2, z2 = y

    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    theta1dot = z1
    z1dot = (m2*g*np.sin(theta2) - m2*s*(L1*z1**2*c + L2*z2**2) -
             (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
    theta2dot = z2
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
             m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
    return theta1dot, z1dot, theta2dot, z2dot

# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 100, 0.001
#tmax, dt = 1000, 0.01
t = np.arange(0, tmax+dt, dt)

for ic in np.arange(10):
    ic_t1 = round(np.random.uniform(0,3),1)
    ic_t2 = round(np.random.uniform(0,3),1)
    ic_z1 = round(np.random.uniform(0,0.2),1)
    ic_z2 = round(np.random.uniform(0,0.2),1)
    y0=[ic_t1,ic_z1,ic_t2,ic_z2]

    
    # Do the numerical integration of the equations of motion
    y = odeint(deriv, y0, t, args=(L1, L2, m1, m2, 9.81))
    # Unpack z and theta as a function of time
    theta1, theta2 = y[:,0], y[:,2]

    print y0
    y0=[ic_t1+1e-9,ic_z1,ic_t2,ic_z2]
    y = odeint(deriv, y0, t, args=(L1, L2, m1, m2, 9.81))
    theta3, theta4 = y[:,0], y[:,2]
    
    # Convert to Cartesian coordinates of the two bob positions.
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    
    x3 = L1 * np.sin(theta3)
    y3 = -L1 * np.cos(theta3)
    x4 = x3 + L2 * np.sin(theta4)
    y4 = y3 - L2 * np.cos(theta4)
    
    lyax = np.log(x3-x1)
    idx = np.isfinite(t[0:2000]) & np.isfinite(lyax[0:2000])
    z=np.polyfit(t[idx],lyax[idx],1)
    print '2000fit ly exp=',round(z[0],4)
    idx = np.isfinite(t[0:10000]) & np.isfinite(lyax[0:10000])
    z=np.polyfit(t[idx],lyax[idx],1)
    print '10000fit ly exp=',round(z[0],4)
    
"""   
for gravity in gs:

    # Initial conditions.
    y0=[0.2,0.0,0.2828,0.0]#ICP1
    y0=[0.05, 0, 0.08, 0.0]#ICP2
    y0=[3.14, 10.0, 3.14, 10.0]#ICP3
    y0=[1.57, 0.0, 0.0, 0.0]#ICQ1
    y0=[1.0, 1.5, 1.0, 1.5]#ICQ2
    y0=[0.2, 0, -0.2, 0.0]#ICQ3
    y0=[0.0, 10.0, 0, 0.0]#ICQ4
    y0=[3.2, 0.5, 3.2, 0.0]#ICC1
    #y0=[2.0, 0.0, 3.14, 0.0]#ICC2
    #y0=[0.0, 10.0, 0, -10.0]#ICC3

    # Do the numerical integration of the equations of motion
    y = odeint(deriv, y0, t, args=(L1, L2, m1, m2, gravity))
    # Unpack z and theta as a function of time
    theta1, theta2 = y[:,0], y[:,2]
    
    
    y0=[0.2+1e-9,0.0,0.2828,0.0]#ICP1
    y0=[0.05+1e-9, 0, 0.08, 0.0]#ICP2
    y0=[3.14+1e-9, 10.0, 3.14, 10.0]#ICP3
    y0=[1.57+1e-9, 0.0, 0.0, 0.0]#ICQ1
    y0=[1.0+1e-9, 1.5, 1.0, 1.5]#ICQ2
    y0=[0.2+1e-9, 0, -0.2, 0.0]#ICQ3
    y0=[0.0+1e-9, 10.0, 0, 0.0]#ICQ4
    y0=[3.2+1e-9, 0.5, 3.2, 0.0]#ICC1
    #y0=[2.0+1e-9, 0.0, 3.14, 0.0]#ICC2
    #y0=[0.0+1e-9, 10.0, 0, -10.0]#ICC3
    
    y = odeint(deriv, y0, t, args=(L1, L2, m1, m2, gravity))
    theta3, theta4 = y[:,0], y[:,2]
    
    #y0 = [0.2, 0.0, 0.2828, 0]
    #y = odeint(deriv, y0, t, args=(L1, L2, m1, m2))
    #theta5, theta6 = y[:,0], y[:,2]
    
    #lya=np.log(theta3-theta1)
    #plt.plot(t,lya)
    
    #plt.figure(2)
    #plt.plot(t,theta1)
    #plt.plot(t,theta3)
    
    
    # Convert to Cartesian coordinates of the two bob positions.
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    
    x3 = L1 * np.sin(theta3)
    y3 = -L1 * np.cos(theta3)
    x4 = x3 + L2 * np.sin(theta4)
    y4 = y3 - L2 * np.cos(theta4)
    
    #x5 = L1 * np.sin(theta5)
    #y5 = -L1 * np.cos(theta5)
    #x6 = x5 + L2 * np.sin(theta6)
    #y6 = y5 - L2 * np.cos(theta6)
    
    #plt.figure(3)
    #plt.plot(t,x1)
    #plt.plot(t,x3)
    
    #lyax = np.log(x3-x1)
    #plt.figure(4)
    #plt.plot(t,lyax)
    
    datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Galatic Dynamics Data\\DoublePendulum\\'
    filename='DoubPen_LsEq1_MsEq1_grav'+str(gravity)+'_ICC1_tstep0p002.npz'
    np.savez(datadir+filename,x1=x1,x2=x2,x3=x3,x4=x4,y1=y1,y2=y2,y3=y3,y4=y4,ic=y0) 

"""
"""
plt.plot(t,theta1)
plt.plot(t,theta2)

# Plotted bob circle radius
r = 0.05
# Plot a trail of the m2 bob's position for the last trail_secs seconds.
trail_secs = 1
# This corresponds to max_trail time points.
max_trail = int(trail_secs / dt)

def make_plot(i):
    # Plot and save an image of the double pendulum configuration for time
    # point i.
    # The pendulum rods.
    ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=2, c='k')
    # Circles representing the anchor point of rod 1, and bobs 1 and 2.
    c0 = Circle((0, 0), r/2, fc='k', zorder=10)
    c1 = Circle((x1[i], y1[i]), r, fc='b', ec='b', zorder=10)
    c2 = Circle((x2[i], y2[i]), r, fc='r', ec='r', zorder=10)
    ax.add_patch(c0)
    ax.add_patch(c1)
    ax.add_patch(c2)

    # The trail will be divided into ns segments and plotted as a fading line.
    ns = 20
    s = max_trail // ns

    for j in range(ns):
        imin = i - (ns-j)*s
        if imin < 0:
            continue
        imax = imin + s + 1
        # The fading looks better if we square the fractional length along the
        # trail.
        alpha = (j/ns)**2
        ax.plot(x2[imin:imax], y2[imin:imax], c='r', solid_capstyle='butt',
                lw=2, alpha=alpha)

    # Centre the image on the fixed anchor point, and ensure the axes are equal
    ax.set_xlim(-L1-L2-r, L1+L2+r)
    ax.set_ylim(-L1-L2-r, L1+L2+r)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig('frames/_img{:04d}.png'.format(i//di))
    plt.cla()


# Make an image every di time points, corresponding to a frame rate of fps
# frames per second.
# Frame rate, s-1
fps = 1
di = int(1.0/fps/dt)
fig, ax = plt.subplots()

for i in range(0, t.size, di):
    print(i // di, '/', t.size // di)
    make_plot(i)
    
"""