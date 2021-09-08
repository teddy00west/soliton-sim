"""
KdV soliton collision simulation

Simulates overtaking collisions, with optional 
dissipation term mu * uxx in the KdV equation.

Stephen Morris, Nov 2018

"""
import numpy as np
from matplotlib import pyplot as plt, cm

import kdv_functions as kdv

# XXX Writes to a directory called kdv_collision_results -- you have to create this

# plot the lines along the peak positions before and after the collision
plot_lines = True

# Set the size of the domain, and create the discretized grid.
L = 50.0
N = 256     # make this a power of 2 to get good FFT derivatives
dx = L / (N - 1.0)
x = np.linspace(0, (1-1.0/N)*L, N)

# Set up the time sample grid.
T = 50      # total time to integrate for
num_t = 501 # number of timesteps
t = np.linspace(0, T, num_t)

# the KdV dissipation parameter; set to zero for pure KdV
mu = 0.0

# intial conditions:
amp1 = 0.5  # the amplitude of the leftmost soliton; vary this to see the effect.  Should be order 1.
amp2 = 0.25  # the amplitude of the next leftmost soliton; Should be order 1 and smaller than amp1.

pos1 = 0.1 * L  # initial positions of the two solitons; near left end
pos2 = 0.4 * L

# Not exactly the 2 soliton solution, but close enough if the peaks are well
# separated
u0 = kdv.kdv_exact(x-pos1, amp1) + kdv.kdv_exact(x-pos2, amp2)

# solve the KdV equation with this initial condition
print('Solving KdV ...')
sol = kdv.diss_kdv_solution(u0, t, L, mu)

# calculate the shifts according to theory for pure KdV (ie mu=0)

Delta = np.arctanh(np.sqrt(amp2/amp1))
D1 = 2*np.sqrt(2/amp1)*Delta
D2 = - 2*np.sqrt(2/amp2)*Delta  # this is a negative shift

# make a line along the peaks before and after the collision

beta1 = 2*amp1  # the speeds of the two solitons
beta2 = 2*amp2

p1b = pos1 + beta1 * t  # peak positions before collision
p2b = pos2 + beta2 * t

p1a = pos1 + beta1 * t + D1  # peak positions after collision
p2a = pos2 + beta2 * t + D2

# Generate a space time plot, with t=0 at the bottom

print('Plotting spacetime ...')
f = plt.figure()
# [::-1, :] is a python trick for reversing the first index so t=0 is at the bottom of the image
plt.imshow(sol[::-1, :], cmap=cm.jet, extent=[0, L, 0, T], zorder=1)
if plot_lines:
    plt.plot(p1b,t, 'w--', linewidth=1, zorder=2)  # white dashed line
    plt.plot(p2b,t, 'w--', linewidth=1, zorder=3)
    plt.plot(p1a,t, 'b--', linewidth=1, zorder=2)  # blue dashed line
    plt.plot(p2a,t, 'b--', linewidth=1, zorder=3)
    plt.xlim(0,L)
    plt.ylim(0,T)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.title(r'KdV for $\eta_{01}$ = '+str(amp1)+', $\eta_{02}$ = '+str(amp2)+', $\mu$ = '+str(mu))
plt.savefig('./kdv_collision_results/kdv_collision_spacetime_lines_amp_'+str(amp1)+' amp2 = '+str(amp2)+' mu = '+str(mu)+'.png')
plt.show()
plt.close(f)
