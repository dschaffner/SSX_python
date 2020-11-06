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
    #print theta1, z1, theta2, z2
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
"""
for ic in np.arange(10):
    ic_t1 = round(np.random.uniform(0,3.2),1)
    ic_t2 = round(np.random.uniform(0,3.2),1)
    ic_z1 = round(np.random.uniform(0,1),1)
    ic_z2 = round(np.random.uniform(0,1),1)
    y0=[ic_t1,ic_z1,ic_t2,ic_z2]
    print y0
    # Do the numerical integration of the equations of motion
    y = odeint(deriv, y0, t, args=(L1, L2, m1, m2, 9.81))
    # Unpack z and theta as a function of time
    theta1, theta2 = y[:,0], y[:,2]

    y0=[ic_t1+1e-9,ic_z1+1e-9,ic_t2+1e-9,ic_z2+1e-9]
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
    idx = np.isfinite(t[0:20000]) & np.isfinite(lyax[0:20000])
    z=np.polyfit(t[idx],lyax[idx],1)
    
    print 'ly exp=',round(z[0],4)
    
"""   


# Initial conditions.
#y0=[3.1, 20.0, 0.1, 20.1]#ICP1
y0=[3.1, 0.2, 0.1, 0.1]#ICscan #28
# Do the numerical integration of the equations of motion
y = odeint(deriv, y0, t, args=(L1, L2, m1, m2, 9.81))
# Unpack z and theta as a function of time
theta1, theta2 = y[:,0], y[:,2]
y0[0]=y0[0]+1e-9#ICP1
print (y0)
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

plt.figure(3)
plt.plot(t[0:100000],x1[0:100000])
plt.plot(t[0:100000],x3[0:100000])

lyax = np.log(x3-x1)
lyax = np.log(abs(x3-x1)/1e-9)
lmax = np.log(0.2/1e-9)
plt.figure(4)
plt.plot(t,lyax)
fitlength=5000
fittime=fitlength*dt
#idx = np.isfinite(t[0:fitlength]) & np.isfinite(lyax[0:fitlength])#clean up NaNs in lyax array
z=np.polyfit(t[0:fitlength],lyax[0:fitlength],1)
plt.plot(t[0:fitlength],z[1]+t[0:fitlength]*z[0],color='red',label='lyapunov exponent = '+str(round(z[0],4)/fittime))
#idx = np.isfinite(t[0:7000]) & np.isfinite(lyax[0:8000])#clean up NaNs in lyax array
#z=np.polyfit(t[idx],lyax[idx],1)
#plt.plot(t[idx],z[1]+t[idx]*z[0],color='green',label='lyapunov exponent = '+str(round(z[0],4)))
plt.legend(loc='lower right',fontsize=12,frameon=False,handlelength=5)
#plt.xlim(-20,120)
#plt.ylim(-10,20)
#plt.hlines(lmax,0,100,color='red')

    
datadir = 'C:\\Users\\dschaffner\\Dropbox\\From OneDrive\\Galatic Dynamics Data\\DoublePendulum\\'
#filename='DoubPen_LsEq1_MsEq1_grav'+str(gravity)+'_ICC1_tstep0p001.npz'
filename='DoubPen_LsEq1_MsEq1_g9p81_tstep001_icscanIC33.npz'
np.savez(datadir+filename,x1=x1,x2=x2,x3=x3,x4=x4,y1=y1,y2=y2,y3=y3,y4=y4,ic=y0) 


