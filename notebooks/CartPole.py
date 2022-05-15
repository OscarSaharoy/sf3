
"""
fork from python-rl and pybrain for visualization
"""

import autograd.numpy as np
import matplotlib.pyplot as plt

# If theta  has gone past our conceptual limits of [-pi,pi]
# map it onto the equivalent angle that is in the accepted range (by adding or subtracting 2pi)
     
def remap_angle(theta):
    
    while theta < -np.pi:
        theta += 2. * np.pi
    while theta > np.pi:
        theta -= 2. * np.pi
   
    return theta
    

## loss function given a state vector. the elements of the state vector are
## [cart location, cart velocity, pole angle, pole angular velocity]

def loss(state):
    return 1-np.exp(-np.dot(state,state)/(2.0 * 0.5**2))

class CartPole:
    """Cart Pole environment. This implementation allows multiple poles,
    noisy action, and random starts. It has been checked repeatedly for
    'correctness', specifically the direction of gravity. Some implementations of
    cart pole on the internet have the gravity constant inverted. The way to check is to
    limit the force to be zero, start from a valid random start state and watch how long
    it takes for the pole to fall. If the pole falls almost immediately, you're all set. If it takes
    tens or hundreds of steps then you have gravity inverted. It will tend to still fall because
    of round off errors that cause the oscillations to grow until it eventually falls.
    """

    def __init__(self, visual=False):
        
        self.reset() 
        
        self.visual = visual

        # Setup pole lengths and masses based on scale of each pole
        # (Papers using multi-poles tend to have them either same lengths/masses
        # or they vary by some scalar from the other poles)
        self.pole_length = 0.5 
        self.pole_mass = 0.5 

        self.mu_c = 0.001 # friction coefficient of the cart
        self.mu_p = 0.001 # friction coefficient of the pole
        self.sim_steps = 12 # number of Euler integration steps to perform in one go
        self.delta_time = 0.045 # time step of the Euler integrator 
        self.max_force = 20.
        self.gravity = 9.8
        self.cart_mass = 0.5

        # for plotting
        self.cartwidth = 1.0
        self.cartheight = 0.2
    
        if self.visual:
            self.drawPlot()
 

    # reset the state vector to the initial state (down-hanging pole)
    def reset(self):
        self.cart_location = 0.0
        self.cart_velocity = 3.0
        self.pole_angle    = np.pi
        self.pole_angvel   = 16.0

            
    def set_state(self, state):
        
        self.cart_location, self.cart_velocity, self.pole_angle, self.pole_angvel = state
           
            
    def get_state(self):

        return np.array([ self.cart_location, self.cart_velocity, self.pole_angle, self.pole_angvel ])

    
    def remap_angle(self):
        
        self.pole_angle = remap_angle( self.pole_angle )
   

    # the loss function that the policy will try to optimise (lower) as a member function
    def loss(self):
        return loss(self.getState())
    
        
    # This is where the equations of motion are implemented
    def perform_action(self, action = 0.0):
        
        # prevent the force from being too large
        force = self.max_force * np.tanh(action/self.max_force)

        # Update state variables
        dt = self.delta_time / float(self.sim_steps)
 
        # integrate forward the equations of motion using the Euler method
        for step in range(self.sim_steps):

            s = np.sin(self.pole_angle)
            c = np.cos(self.pole_angle)
            
            m = 4.0 * ( self.cart_mass + self.pole_mass ) - 3.0 * self.pole_mass * c**2
            
            cart_accel = 1/m * ( 
                2.0 * ( self.pole_length * self.pole_mass * s * self.pole_angvel**2
                + 2.0 * ( force -self.mu_c * self.cart_velocity ) )
                - 3.0 * self.pole_mass * self.gravity * c*s 
                + 6.0 * self.mu_p * self.pole_angvel * c / self.pole_length
            ) 
            
            pole_accel = 1/m * (
                - 1.5*c / self.pole_length * ( 
                    self.pole_length / 2.0 * self.pole_mass * s * self.pole_angvel**2 
                    + force 
                    - self.mu_c * self.cart_velocity
                )
                + 6.0 * ( self.cart_mass + self.pole_mass ) / ( self.pole_mass * self.pole_length ) * \
                ( self.pole_mass * self.gravity * s - 2.0/self.pole_length * self.mu_p * self.pole_angvel )
            )
           
            # Do the updates in this order, so that we get semi-implicit Euler that is simplectic rather than forward-Euler which is not. 
            self.cart_velocity += dt * cart_accel
            self.pole_angvel   += dt * pole_accel
            self.pole_angle    += dt * self.pole_angvel
            self.cart_location += dt * self.cart_velocity

        if self.visual:
            self._render()


   # the following are graphics routines
    def drawPlot(self):
        
        plt.ion()
        self.fig = plt.figure( figsize=(9.5,2) )
        
        # draw cart
        self.axes = self.fig.add_subplot(111, aspect='equal')
        self.box = plt.Rectangle(xy=(self.cart_location - self.cartwidth / 2.0, -self.cartheight / 2.0), 
                             width=self.cartwidth, height=self.cartheight)
        self.axes.add_artist(self.box)
        self.box.set_clip_box(self.axes.bbox)

        # draw pole
        self.pole = plt.Line2D([self.cart_location, self.cart_location + np.sin(self.pole_angle) * self.pole_length], 
                           [0, np.cos(self.pole_angle) * self.pole_length], linewidth=3.5, color='black')
        self.axes.add_artist(self.pole)
        self.pole.set_clip_box(self.axes.bbox)

        # set axes limits
        self.axes.set_xlim(-10, 10)
        self.axes.set_ylim(-1, 1)
        self.fig.tight_layout()
 
        
    def _render(self):
        
        self.box.set_x(self.cart_location - self.cartwidth / 2.0)
        self.pole.set_xdata([ self.cart_location, self.cart_location + np.sin(self.pole_angle) * self.pole_length ])
        self.pole.set_ydata([ 0, np.cos(self.pole_angle) * self.pole_length ])

        