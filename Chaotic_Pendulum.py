#!/usr/bin/env python
# coding: utf-8

# - Dmitri LaBelle 
# 

# In[4]:


import sys, math
import numpy as np
import matplotlib.pyplot as plt

# Global parameters.

alphaList = 0.5
Glist = 1.15
WD = 2/3 #frequency - Driving force
X0 = 1.0
V0 = 1
K  = 1.0
DT = 0.001
TMAX = 3500.0

#-------------------------------------------------------------------------

def acc(x, v, t):
    return -K*math.sin(x) - alpha * v  + G*math.cos(WD*t) #Driving force

def potential(x):
    return K

def energy(x, v):
    return potential(x) + 0.5*v*v

#-------------------------------------------------------------------------

def interp(x0, y0, x1, y1, x):
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

def output(x, v, t):
    print (t, x, v, energy(x, v) - E0)
    
'''def check(phi_current, phi_previous):
    for j in np.arange(phi_current, phi_previous, 0.0001):
        if j % ((2 * math.pi)) == 0:'''
    

def time_to_stop(x, v, t, tmax):
    
    # Stopping condition.
    
    if t > tmax:
        return 1
    else:
        return 0

def take_a_step(x, v, t, dt):
    
    # Take a single time step.
    
    a = acc(x, v, t)
    
    # Prediction:
    x += v*dt + 0.5*a*dt*dt
    v += a*dt
    t += dt
    
    # Correction:
    a1 = acc(x, v, t)
    v += 0.5*dt*(a1-a)

    return x,v,t


alpha = alphaList
G = Glist
WD = WD
xponplot = []
xplot = []
tplot = [] 
vponplot = []
vplot = []

t = 0
tp = t
tpp = tp
x = X0
xp = x
xpp = xp
v = V0
vp = v
vpp = vp
tmax = TMAX
dt = DT

phi = 0
phip = 0
phipp = 0

E0 = energy(x, v)

j = 1


while time_to_stop(x, v, t, tmax) == 0:
    phipp = phip
    phip = phi
    tpp = tp
    tp = t
    vpp = vp
    vp = v
    xpp = xp
    xp = x

    (x,v,t) = take_a_step(x, v, t, dt)
    
    phi = t * WD
    
    #var = (tmax * WD)/ (2*math.pi)
    #var = round(var)
    #print(phi, phip)

    while (x > math.pi):
        x -= 2*math.pi
    while (x < -math.pi):
        x += 2*math.pi

    if phi > (2 * j * math.pi) > phip or phi < (2 * j * math.pi) < phip:
        #print('hewwo')
        #print(j)
        xponplot.append(interp(vpp, xpp, vp, xp, v))
        vponplot.append(interp(xpp, vpp, xp, vp, x))
        j = j + 1
    
    if t >= 500:
        xplot.append(x)
        vplot.append(v)

            #output(x, v, t)
print("Plot for G =", G, 'And alpha =', alpha)

plt.plot(xplot, vplot)
plt.plot(xponplot, vponplot, 'ro')
plt.xlabel('position')
plt.ylabel('velocity')
plt.axis([-3.5,3.5,-3,3])
plt.plot
plt.show()


# In[8]:


import sys, math
import matplotlib.pyplot as plt

# Global parameters.

alphaList = [0.0, 0.5, 0.5, 0.5, 0.5, 0.5]
Glist = [0.0, 0.0, 1, 1.07, 1.15, 1.5]#amplitude - Driving force
WD = 2/3 #frequency - Driving force
X0 = 1.0
V0 = 1
K  = 1.0
DT = 0.001
TMAX = 3500.0

#-------------------------------------------------------------------------

def acc(x, v, t):
    return -K*math.sin(x) - alpha * v  + G*math.cos(WD*t) #Driving force

def potential(x):
    return K

def energy(x, v):
    return potential(x) + 0.5*v*v

#-------------------------------------------------------------------------

def interp(x0, y0, x1, y1, x):
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

def output(x, v, t):
    print (t, x, v, energy(x, v) - E0)

def time_to_stop(x, v, t, tmax):
    
    # Stopping condition.
    
    if t > tmax:
        return 1
    else:
        return 0

def take_a_step(x, v, t, dt):
    
    # Take a single time step.
    
    a = acc(x, v, t)
    
    # Prediction:
    x += v*dt + 0.5*a*dt*dt
    v += a*dt
    t += dt
    
    # Correction:
    a1 = acc(x, v, t)
    v += 0.5*dt*(a1-a)

    return x,v,t

for i in range (0,len(Glist)):
    alpha = alphaList[i]
    G = Glist[i]
    WD = WD
    xplot = []
    tplot = [] 
    vplot = []
    xponplot = []
    vponplot = []
    
    phi = 0
    phip = 0
    phipp = 0


    t = 0
    tp = t
    tpp = tp
    x = X0
    xp = x
    xpp = xp
    v = V0
    vp = v
    vpp = vp
    tmax = TMAX
    dt = DT

    E0 = energy(x, v)
    
    j = 1
    

    while time_to_stop(x, v, t, tmax) == 0:
        phipp = phip
        phip = phi
        tpp = tp
        tp = t
        xpp = xp
        xp = x

        (x,v,t) = take_a_step(x, v, t, dt)
        
        phi = t * WD
        
        while (x > math.pi):
            x -= 2*math.pi
        while (x < -math.pi):
            x += 2*math.pi
            
        if phi > (2 * j * math.pi) > phip or phi < (2 * j * math.pi) < phip:
            #print('hewwo')
            #print(j)
            xponplot.append(x)
            vponplot.append(v)
            j = j + 1
        
        if t >= 250:
            xplot.append(x)
            tplot.append(t)
            vplot.append(v)

        #output(x, v, t)
    print("Plot for G =", G, 'And alpha =', alpha)
    
    plt.plot(xplot, vplot)
    plt.plot(xponplot, vponplot, 'ro')
    plt.xlabel('position')
    plt.ylabel('velocity')
    plt.axis([-3.5,3.5,-3,3]) #note for (G, Alpha) = (0.0, 0.5), due to the confining condictions of this line of code, this graph does not appear. Data is collected, but the results are too small to see on this scale. 
    plt.plot
    plt.show()


# In[11]:


import sys, math
import numpy as np
import matplotlib.pyplot as plt

# Global parameters.

alphaList = 0.5
Glist = 1.15
WD = 2/3 #frequency - Driving force
X0 = 1.0
V0 = 1
K  = 1.0
DT = 0.001
TMAX = 3500.0

#-------------------------------------------------------------------------

def acc(x, v, t):
    return -K*math.sin(x) - alpha * v  + G*math.cos(WD*t) #Driving force

def potential(x):
    return K

def energy(x, v):
    return potential(x) + 0.5*v*v

#-------------------------------------------------------------------------

def interp(x0, y0, x1, y1, x):
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

def output(x, v, t):
    print (t, x, v, energy(x, v) - E0)
    
'''def check(phi_current, phi_previous):
    for j in np.arange(phi_current, phi_previous, 0.0001):
        if j % ((2 * math.pi)) == 0:'''
    

def time_to_stop(x, v, t, tmax):
    
    # Stopping condition.
    
    if t > tmax:
        return 1
    else:
        return 0

def take_a_step(x, v, t, dt):
    
    # Take a single time step.
    
    a = acc(x, v, t)
    
    # Prediction:
    x += v*dt + 0.5*a*dt*dt
    v += a*dt
    t += dt
    
    # Correction:
    a1 = acc(x, v, t)
    v += 0.5*dt*(a1-a)

    return x,v,t


alpha = alphaList
G = Glist
WD = WD
xponplot = []
xponplot1 = []
xplot = []
tplot = [] 
vponplot = []
vponplot1 = []
vplot = []

t = 0
tp = t
tpp = tp
x = X0
xp = x
xpp = xp
v = V0
vp = v
vpp = vp
tmax = TMAX
dt = DT

phi = 0
phip = 0
phipp = 0

E0 = energy(x, v)

j = 1
k = 1


while time_to_stop(x, v, t, tmax) == 0:
    phipp = phip
    phip = phi
    tpp = tp
    tp = t
    xpp = xp
    xp = x

    (x,v,t) = take_a_step(x, v, t, dt)
    
    phi = t * WD
    
    #var = (tmax * WD)/ (2*math.pi)
    #var = round(var)
    #print(phi, phip)

    while (x > math.pi):
        x -= 2*math.pi
    while (x < -math.pi):
        x += 2*math.pi

    if phi > ((2 * j * math.pi)+ (math.pi/4)) > phip or phi < ((2 * j * math.pi)+ (math.pi/4)) < phip:
        #print('hewwo')
        #print(j)
        xponplot.append(x)
        vponplot.append(v)
        j = j + 1
        
    if phi > ((2 * k * math.pi)+ (math.pi/2)) > phip or phi < ((2 * k * math.pi)+ (math.pi/2)) < phip:
        #print('hewwo')
        #print(j)
        xponplot1.append(x)
        vponplot1.append(v)
        k = k + 1    
    
    
    if t >= 500:
        xplot.append(x)
        vplot.append(v)

            #output(x, v, t)
print("Plot for G =", G, 'And alpha =', alpha)

plt.plot(xplot, vplot)
plt.plot(xponplot, vponplot, 'ro', label = '2npi + pi/4')
plt.plot(xponplot1, vponplot1, 'y^', label = '2npi + pi/2')
plt.legend()
plt.legend(bbox_to_anchor=(1,1))
plt.xlabel('position')
plt.ylabel('velocity')
plt.axis([-3.5,3.5,-3,3])
plt.plot
plt.show()

plt.plot(xplot, vplot)
plt.plot(xponplot, vponplot, 'ro', label = '2npi + pi/4')
plt.plot(xponplot1, vponplot1, 'y^', label = '2npi + pi/2')
plt.legend()
plt.legend(bbox_to_anchor=(1,1))
plt.xlabel('position')
plt.ylabel('velocity')
plt.axis([1.5,3,0,2])
plt.plot
plt.show()

plt.plot(xplot, vplot)
plt.plot(xponplot, vponplot, 'ro', label = '2npi + pi/4')
plt.plot(xponplot1, vponplot1, 'y^', label = '2npi + pi/2')
plt.legend()
plt.legend(bbox_to_anchor=(1,1))
plt.xlabel('position')
plt.ylabel('velocity')
plt.axis([-3.2,-2.5,0.5,1.5])
plt.plot
plt.show()


# In[ ]:




