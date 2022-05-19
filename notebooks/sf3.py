#!/usr/bin/env python
# coding: utf-8

# import CartPole.py from local directory
import CartPole, sf3utility
import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import random

plt.rcParams["font.family"] = "Georgia"
#plt.rcParams['figure.figsize'] = [9.0, 7.0]
#plt.rcParams['figure.dpi'] = 400

# store results for later
cache = {}


# allows nice plots that can be redrawn
get_ipython().run_line_magic('matplotlib', 'notebook')


# # Task 1.1 - Simulation of Rollout
# instantiate a cartpole and use a small timestep to get smooth lines
rollout_cartpole = CartPole.CartPole()

rollout_cartpole.sim_steps = 1
rollout_cartpole.delta_time = 0.01


# ## Small Oscillation about Stable Equilibirum
rollout_cartpole.reset()

rollout_cartpole.cart_velocity = 0.126
rollout_cartpole.pole_angvel = 1

x, y = [], []
states = []

for i in range(50):
    
    rollout_cartpole.perform_action()
    
    x.append( rollout_cartpole.cart_velocity )
    y.append( rollout_cartpole.pole_angvel )
    
    states.append( rollout_cartpole.get_state() )
    
    
fig, ax = plt.subplots(1, 1, num=2)
sf3utility.setup_phase_portrait( ax )
ax.plot( x, y )

ax.set_title( "Small Oscillation about Stable Equilibirum" )
ax.set_xlabel( "Cart Velocity" )
ax.set_ylabel( "Pole Angular Velocity" )

states = np.array( states )
cache["states_small_oscillations"] = states


# ## Large Amplitude Oscillations

# large oscillations about stable equilibrium

rollout_cartpole.reset()

rollout_cartpole.cart_velocity = 1.72
rollout_cartpole.pole_angvel = 12
rollout_cartpole.mu_p = 0.001 # increase friction to speed up convergence

x, y = [], []
states = []

for i in range(10000):
    
    rollout_cartpole.perform_action()
    
    x.append( rollout_cartpole.cart_velocity )
    y.append( rollout_cartpole.pole_angvel )
    
    states.append( rollout_cartpole.get_state() )
    
    
fig, ax = plt.subplots(1, 1, num=3)
sf3utility.setup_phase_portrait( ax )

ax.set_xlim( min(x) * 1.12, max(x) * 1.12 )
ax.set_ylim( min(y) * 1.12, max(y) * 1.12 )

points   = np.array([x, y]).T.reshape(-1, 1, 2)[::-1]
segments = np.concatenate( [points[:-1], points[1:]], axis=1 )
colouring_array =  np.linspace( 0.0, 1.0, len(x) ) ** 3

linecollection = matplotlib.collections.LineCollection( segments, array=colouring_array, cmap="cool", zorder=3, linewidths=1.2 )

ax.add_collection( linecollection )

ax.set_title( "Large Oscillations of the Cartpole System" )
ax.set_xlabel( "Cart Velocity" )
ax.set_ylabel( "Pole Angular Velocity" )

states = np.array( states )
cache["states_large_oscillations"] = states


# ## Swinging Over the Top

# even larger oscillations

rollout_cartpole.reset()

rollout_cartpole.cart_velocity = 4
rollout_cartpole.pole_angvel = 20
rollout_cartpole.mu_p = 0.005 # increase friction to speed up convergence
    
x, y = [], []

for i in range( 5700 ):
    
    pole_angle = rollout_cartpole.pole_angle % (2*np.pi)
    
    if pole_angle > 6.1:
        x.append( np.nan )
        y.append( np.nan )
        
    else:
        x.append( pole_angle )
        y.append( rollout_cartpole.pole_angvel )
    
    rollout_cartpole.perform_action()
    
    
fig, ax = plt.subplots(1, 1, num=4)
sf3utility.setup_phase_portrait( ax )
ax.axvline( x=np.pi, color="black" )

ax.set_xlim( min(x) * 1.12, max(x) * 1.12 )
ax.set_ylim( min(y) * 1.12, max(y) * 1.12 )

points   = np.array([x, y]).T.reshape(-1, 1, 2)[::-1]
segments = np.concatenate( [points[:-1], points[1:]], axis=1 )
colouring_array =  np.linspace( 0.0, 1.0, len(x) ) ** 3

linecollection = matplotlib.collections.LineCollection( segments, array=colouring_array, cmap="cool", zorder=3 )

ax.add_collection( linecollection )

ax.set_xlim(0.2, 6)
ax.set_xticks( np.pi * np.array([0.5, 1, 1.5]) )
ax.set_xticklabels( ["π/2", "π", "3π/2"] )

ax.set_title( "Phase Plot with Complete Rotations" )
ax.set_xlabel( "Pole Angle" )
ax.set_ylabel( "Pole Angular Velocity" )


# ## Effect of Varying Initial Cart Velocity and Pole Angle

# sweep over different initial cart velocities and angles and find the subsequent change in state

# number of steps to vary the intial conditions across their range
Nsteps = 30

# setup some intial conditions to loop over, varying the intial pole angle and cart velocity

initial_cart_positions  = np.array( [1] )
initial_cart_velocities = np.linspace( -10, 10, num=Nsteps )
initial_pole_angles     = np.linspace( -np.pi, np.pi, num=Nsteps )
initial_pole_angvels    = np.array( [0] )

# create array of initial state vectors

initial_states = np.array( np.meshgrid( 
    initial_cart_positions, 
    initial_cart_velocities, 
    initial_pole_angles, 
    initial_pole_angvels 
)).T.squeeze()

# get 2d array of subsquent state vectors

subsequent_states = [ CartPole.perform_action( state ) for state in initial_states.reshape( (Nsteps**2,4) ) ]
subsequent_states = np.array( subsequent_states ).reshape( (Nsteps, Nsteps, 4) )

state_changes = subsequent_states - initial_states


fig, ax = plt.subplots(1, 1, num=5)  

# create array of interpolated colours 

col_lerp = np.linspace(0, 1, Nsteps)[np.newaxis].T
colours = ( 1 - col_lerp ) * np.array( [0, 255, 231, 255] )/255 + col_lerp * np.array( [255, 0, 230, 255] )/255


for i, row in enumerate(state_changes):
       
    # convert to arrays and extract certain components from vectors
    
    x = initial_states[:,i,2] # extract initial angle
    y = state_changes[:,i,0] # extract change in cart position
    
    # code to smooth plot lines
    
    xnew = np.linspace( x.min(), x.max(), 300 ) 

    y_spline = scipy.interpolate.make_interp_spline(x, y, k=2)
    y_smooth = y_spline(xnew)
    
    # plot then move onto next line

    ax.plot( xnew, y_smooth, color=colours[i] ) 

ax.set_title( "Exploration of State Change Function" )
ax.set_xlabel( "Initial Pole Angle" )
ax.set_ylabel( "Subsequent Change in Cart Position" )


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, num=6, figsize=(9,9))

titles = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]

for i, ax in enumerate( [ax1, ax2, ax3, ax4] ):

    ax.imshow( state_changes[:,:,i], interpolation="bicubic", extent=(-10, 10, -np.pi, np.pi), aspect='auto', cmap="cool", origin='lower' )
    contour = ax.contour( initial_states[0,:,1], initial_states[:,0,2], state_changes[:,:,i], colors="white", linewidths=1 )
    ax.clabel( contour, contour.levels[1::2], inline=True, fontsize=12 )
    
    ax.set_title( titles[i] )
    
fig.text(0.5, 0.95, 'Changes in State as a Function of Initial State', ha='center', va='center', fontsize=14)
fig.text(0.5, 0.04, 'Initial Cart Velocity', ha='center', va='center', fontsize=12)
fig.text(0.06, 0.5, 'Initial Pole Angle', ha='center', va='center', rotation='vertical', fontsize=12)


# ## Effect of Varying Initial Pole Angle and Angular Velocity

# sweep over different initial pole angles and angvels and find the subsequent change in state

# number of steps to vary the intial conditions across their range
Nsteps = 30

# setup some intial conditions to loop over, varying the intial pole angle and angular velocity

initial_cart_positions  = np.array( [2] )
initial_cart_velocities = np.array( [4] )
initial_pole_angles     = np.linspace( -np.pi, np.pi, num=Nsteps )
initial_pole_angvels    = np.linspace( -15, 15, num=Nsteps )

# create array of initial state vectors

initial_states = np.array( np.meshgrid(
    initial_cart_positions, 
    initial_cart_velocities, 
    initial_pole_angles, 
    initial_pole_angvels 
)).T.squeeze()

# get 2d array of subsquent state vectors

state_changes = [ CartPole.perform_action( state ) - state for state in initial_states.reshape( (Nsteps**2,4) ) ]
state_changes = np.array( state_changes ).reshape( (Nsteps, Nsteps, 4) )

# cache data for later

cache["state_changes_varying_angle_and_angvel"] = state_changes


fig, ax = plt.subplots(1, 1, num=8)  

# create array of interpolated colours 

col_lerp = np.linspace(0, 1, Nsteps)[np.newaxis].T
colours = ( 1 - col_lerp ) * np.array( [0, 255, 231, 255] )/255 + col_lerp * np.array( [255, 0, 230, 255] )/255


for i, row in enumerate(state_changes):
       
    # convert to arrays and extract certain components from vectors
    
    x = initial_states[:,i,3] # extract initial angular velocity
    y = state_changes[:,i,1] # extract change in cart velocity
    
    # code to smooth plot lines
    
    xnew = np.linspace( x.min(), x.max(), 300 ) 

    y_spline = scipy.interpolate.make_interp_spline(x, y, k=2)
    y_smooth = y_spline(xnew)
    
    # plot then move onto next line

    ax.plot( xnew, y_smooth, color=colours[i] ) 

ax.set_title( "Exploration of State Change Function" )
ax.set_xlabel( "Initial Pole Angular Velocity" )
ax.set_ylabel( "Subsequent Change in Cart Velocity" )


fig, ax = plt.subplots(1, 1, num=9)  

# create array of interpolated colours 

col_lerp = np.linspace(0, 1, Nsteps)[np.newaxis].T
colours = ( 1 - col_lerp ) * np.array( [0, 255, 231, 255] )/255 + col_lerp * np.array( [255, 0, 230, 255] )/255


for i, row in enumerate(state_changes):
       
    # convert to arrays and extract certain components from vectors
    
    x = initial_states[i,:,2] # extract initial angular velocity
    y = state_changes[i,:,1] # extract change in cart velocity
    
    # code to smooth plot lines
    
    xnew = np.linspace( x.min(), x.max(), 300 ) 

    y_spline = scipy.interpolate.make_interp_spline(x, y, k=2)
    y_smooth = y_spline(xnew)
    
    # plot then move onto next line

    ax.plot( xnew, y_smooth, color=colours[i] ) 

ax.set_title( "Exploration of State Change Function" )
ax.set_xlabel( "Initial Pole Angle" )
ax.set_ylabel( "Subsequent Change in Cart Velocity" )


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, num=11, figsize=(9,9))

titles = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]

for i, ax in enumerate( [ax1, ax2, ax3, ax4] ):

    ax.imshow( state_changes[:,:,i], interpolation="bicubic", extent=(-np.pi, np.pi, -15, 15), aspect='auto', cmap="cool", origin='lower' )
    contour = ax.contour( initial_states[0,:,2], initial_states[:,0,3], state_changes[:,:,i], colors="white", linewidths=1 )
    ax.clabel( contour, contour.levels[1::2], inline=True, fontsize=12 )    
    ax.set_title( titles[i] )
    
fig.text(0.5, 0.95, 'Changes in State as a Function of Initial State', ha='center', va='center', fontsize=14)
fig.text(0.5, 0.04, 'Initial Pole Angle', ha='center', va='center', fontsize=12)
fig.text(0.06, 0.5, 'Initial Pole Angular Velocity', ha='center', va='center', rotation='vertical', fontsize=12)


# # Task 1.3: Linear Model
# ## Generating Training Data

# set the random seed to make this cell deterministic
np.random.seed(4)

# generate random arrays of 500 values

random_positions  = np.random.rand( 500 ) * 10 - 5
random_velocities = np.random.rand( 500 ) * 20 - 10
random_angles     = np.random.rand( 500 ) * np.pi * 2 - np.pi
random_angvels    = np.random.rand( 500 ) * 30 - 15

# stack random values into 500 state vectors

X = initial_states = np.stack( [
    random_positions,
    random_velocities,
    random_angles,
    random_angvels
] ).T

Y = subsequent_states = np.array( [ CartPole.perform_action( state ) - state for state in initial_states ] )


# ## Finding the Least-Squares Fit

Xplus = np.linalg.inv(X.T @ X) @ X.T
C = Xplus @ Y

cache["C_large_deviations"] = C


# ## Evaluating the Linear Estimator
# ## Plotting Predicted State Change Against Target State Change

fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2, num=19, figsize=(9,9))

XC = X @ C

titles = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]

for i, ax in enumerate( [ax1, ax2, ax3, ax4] ):
    
    x, y = (XC)[:,i], Y[:,i]
    c = np.abs(x - y)
    
    extent = np.max( ( np.concatenate([x, y]) ) ) * 1.2
    
    ax.scatter( y, x, s=1, c=c, cmap="cool" )
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    
    ax.plot( [-extent, extent], [-extent, extent], color="black", linestyle="dotted" )
    
    ax.set_title( titles[i] )
    
fig.text(0.5, 0.95, 'Predicted State Changes vs. Target State Changes', ha='center', va='center', fontsize=14)
fig.text(0.5, 0.04, 'Target State Change', ha='center', va='center', fontsize=12)
fig.text(0.06, 0.5, 'Predicted State Change', ha='center', va='center', rotation='vertical', fontsize=12)
    


# ## Linear Model for Small Deflections Only
# generate random arrays of 500 values

random_positions  = np.random.rand( 500 ) * 0.5 - 0.25
random_velocities = np.random.rand( 500 ) * 0.5 - 0.25
random_angles     = np.random.rand( 500 ) * 0.5 - 0.25
random_angvels    = np.random.rand( 500 ) * 0.5 - 0.25

# stack random values into 500 state vectors

X = initial_states = np.stack( [
    random_positions,
    random_velocities,
    random_angles,
    random_angvels
] ).T

Y = subsequent_states = np.array( [ CartPole.perform_action( state ) - state for state in initial_states ] )


Xplus = np.linalg.inv(X.T @ X) @ X.T
C = Xplus @ Y


fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2, num=20, figsize=(9,9))

XC = X @ C

titles = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]

for i, ax in enumerate( [ax1, ax2, ax3, ax4] ):
    
    x, y = (XC)[:,i], Y[:,i]
    c = np.abs(x - y)
    
    extent = np.max( ( np.concatenate([x, y]) ) ) * 1.2
    
    ax.scatter( y, x, s=1, c=c, cmap="cool" )
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    
    ax.plot( [-extent, extent], [-extent, extent], color="black", linestyle="dotted" )
    
    ax.set_title( titles[i] )
    
fig.text(0.5, 0.95, 'Predicted State Changes vs. Target State Changes', ha='center', va='center', fontsize=14)
fig.text(0.5, 0.04, 'Target State Change', ha='center', va='center', fontsize=12)
fig.text(0.06, 0.5, 'Predicted State Change', ha='center', va='center', rotation='vertical', fontsize=12)


# ## Comparing Prediction Contour Plots to Target Contour Plots

C = cache["C_large_deviations"]

# sweep over different initial pole angles and angvels and find the predicted change in state

# number of steps to vary the intial conditions across their range
Nsteps = 30

# setup some intial conditions to loop over, varying the intial pole angle and angular velocity

initial_cart_positions  = np.array( [2] )
initial_cart_velocities = np.array( [4] )
initial_pole_angles     = np.linspace( -np.pi, np.pi, num=Nsteps )
initial_pole_angvels    = np.linspace( -15, 15, num=Nsteps )

# create array of initial state vectors

initial_states = np.array( np.meshgrid(
    initial_cart_positions, 
    initial_cart_velocities, 
    initial_pole_angles, 
    initial_pole_angvels 
)).T.squeeze()

# get 2d array of subsquent state vectors

predicted_changes = initial_states.reshape( (Nsteps**2,4) ) @ C
predicted_changes = predicted_changes.reshape( (Nsteps, Nsteps, 4) )

cache["linear_predicted_changes_varying_angle_and_angvel"] = predicted_changes


state_changes = cache["state_changes_varying_angle_and_angvel"]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, num=15, figsize=(9,9))

titles = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]

for i, ax in enumerate( [ax1, ax2, ax3, ax4] ):

    ax.imshow( predicted_changes[:,:,i], interpolation="bicubic", extent=(-np.pi, np.pi, -15, 15), aspect='auto', cmap="cool", origin='lower' )
    target_contour = ax.contour( initial_states[0,:,2], initial_states[:,0,3], state_changes[:,:,i], colors="black", linewidths=1 )
    
    ax.clabel( target_contour, target_contour.levels[1::2], inline=True, fontsize=12 ) 
    estimate_contour = ax.contour( initial_states[0,:,2], initial_states[:,0,3], predicted_changes[:,:,i], colors="white", linewidths=1 )
    
    ax.clabel( estimate_contour, estimate_contour.levels[1::2], inline=True, fontsize=12 )    
    ax.set_title( titles[i] )
    
fig.text(0.5, 0.95, 'Linear Estimate of State Change Compared to Target', ha='center', va='center', fontsize=14)
fig.text(0.5, 0.04, 'Initial Pole Angle', ha='center', va='center', fontsize=12)
fig.text(0.06, 0.5, 'Initial Pole Angular Velocity', ha='center', va='center', rotation='vertical', fontsize=12)


# sweep over different initial cart velocities and angles and find the subsequent change in state

# number of steps to vary the intial conditions across their range
Nsteps = 30

# setup some intial conditions to loop over, varying the intial cart velocity and pole angular velocity

initial_cart_positions  = np.array( [1] )
initial_cart_velocities = np.linspace( -10, 10, num=Nsteps )
initial_pole_angles     = np.array( [2] )
initial_pole_angvels    = np.linspace( -15, 15, num=Nsteps )

# create array of initial state vectors

initial_states = np.array( np.meshgrid( 
    initial_cart_positions, 
    initial_cart_velocities, 
    initial_pole_angles, 
    initial_pole_angvels 
)).T.squeeze()

# get 2d array of subsquent state vectors

state_changes = [ CartPole.perform_action( state ) - state for state in initial_states.reshape( (Nsteps**2,4) ) ]
state_changes = np.array( state_changes ).reshape( (Nsteps, Nsteps, 4) )

predicted_changes = initial_states.reshape( (Nsteps**2,4) ) @ C
predicted_changes = np.array( predicted_changes ).reshape( (Nsteps, Nsteps, 4) )


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, num=16, figsize=(9,9))

titles = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]

for i, ax in enumerate( [ax1, ax2, ax3, ax4] ):

    ax.imshow( predicted_changes[:,:,i], interpolation="bicubic", extent=(-10, 10, -15, 15), aspect='auto', cmap="cool", origin='lower' )
    target_contour = ax.contour( initial_states[0,:,1], initial_states[:,0,3], state_changes[:,:,i], colors="black", linewidths=1 )
    
    ax.clabel( target_contour, inline=True, fontsize=8 ) 
    estimate_contour = ax.contour( initial_states[0,:,1], initial_states[:,0,3], predicted_changes[:,:,i], colors="white", linewidths=1 )
    
    ax.clabel( estimate_contour, inline=True, fontsize=8 )    
    ax.set_title( titles[i] )
    
fig.text(0.5, 0.95, 'Linear Estimate of State Change Compared to Target', ha='center', va='center', fontsize=14)
fig.text(0.5, 0.04, 'Initial Cart Velocity', ha='center', va='center', fontsize=12)
fig.text(0.06, 0.5, 'Initial Pole Angular Velocity', ha='center', va='center', rotation='vertical', fontsize=12)


# # Task 1.4: Prediction of System Evolution

# ## Small Oscillations

actual_states = cache["states_small_oscillations"]

fig, ax = plt.subplots(1, 1, num=22)
sf3utility.setup_phase_portrait( ax )

# small oscillations about stable equilibrium

state = np.array( [0, 0.126, np.pi, 1] )

prediction_states = []

for i in range(80):
    
    prediction_states.append( state )
    state = state @ ( C + np.identity(4) )
    state[2] = CartPole.remap_angle( state[2] )
    

prediction_states = np.array( prediction_states )
    
x = prediction_states[:,1]
y = prediction_states[:,3]

f, u = scipy.interpolate.splprep( [x, y], s=0, per=True )
xint, yint = scipy.interpolate.splev(np.linspace(0, 1, 10000), f)

ax.set_xlim( -1.3, 3 )
ax.set_ylim( -7, 9 )

points   = np.array([xint, yint]).T.reshape(-1, 1, 2)[::-1]
segments = np.concatenate( [points[:-1], points[1:]], axis=1 )[500:]
colouring_array =  np.linspace( 0.0, 1.0, len(xint) )# ** 3

linecollection = matplotlib.collections.LineCollection( segments, array=colouring_array, cmap="cool", zorder=3 )

ax.add_collection( linecollection )

x = actual_states[:,1]
y = actual_states[:,3]

ax.plot( x, y, color="orange", linewidth=3 )

ax.set_title( "Small Oscillation about Supposedly Stable Equilibirum" )
ax.set_xlabel( "Cart Velocity" )
ax.set_ylabel( "Pole Angular Velocity" )


actual_states = cache["states_large_oscillations"]

# large oscillations about stable equilibrium

state = np.array( [0, 1.72, np.pi, 10] )

x, y = [], []

for i in range(1000):
    
    x.append( state[1] )
    y.append( state[3] )
    
    state += state @ C
    state[2] = CartPole.remap_angle( state[2] )
    
    
fig, ax = plt.subplots(1, 1, num=23)
sf3utility.setup_phase_portrait( ax )

ax.set_xlim( min(x) * 1.12, max(x) * 1.12 )
ax.set_ylim( min(y) * 1.12, max(y) * 1.12 )

points   = np.array([x, y]).T.reshape(-1, 1, 2)[::-1]
segments = np.concatenate( [points[:-1], points[1:]], axis=1 )
colouring_array =  np.linspace( 0.0, 1.0, len(x) )

linecollection = matplotlib.collections.LineCollection( segments, array=colouring_array, cmap="cool", zorder=3 )

ax.add_collection( linecollection )

ax.set_title( "Large Oscillations in the Linear Model" )
ax.set_xlabel( "Cart Velocity" )
ax.set_ylabel( "Pole Angular Velocity" )

states = np.array( states )


# even larger oscillations

rollout_cartpole.reset()

state = np.array( [0, 4, np.pi, 20] )
    
x, y = [], []

for i in range( 300 ):
    
    pole_angle = state[2] % (2*np.pi)
    
  
    x.append( pole_angle )
    y.append( state[3] )
    
    state += state @ C
    state[2] = CartPole.remap_angle( state[2] )
    
    
fig, ax = plt.subplots(1, 1, num=24)
sf3utility.setup_phase_portrait( ax )
ax.axvline( x=np.pi, color="black" )

ax.set_xlim( min(x) * 1.12, max(x) * 1.12 )
ax.set_ylim( min(y) * 1.12, max(y) * 1.12 )

points   = np.array([x, y]).T.reshape(-1, 1, 2)[::-1]
segments = np.concatenate( [points[:-1], points[1:]], axis=1 )
colouring_array =  np.linspace( 0.0, 1.0, len(x) ) ** 3

linecollection = matplotlib.collections.LineCollection( segments, array=colouring_array, cmap="cool", zorder=3 )

ax.add_collection( linecollection )

ax.set_xlim(0.2, 6)
ax.set_xticks( np.pi * np.array([0.5, 1, 1.5]) )
ax.set_xticklabels( ["π/2", "π", "3π/2"] )

ax.set_title( "Phase Plot with Complete Rotations" )
ax.set_xlabel( "Pole Angle" )
ax.set_ylabel( "Pole Angular Velocity" )


