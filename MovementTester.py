"""
This is a mutilated version of PendulumEnv.py, used to test the movement of the pendulum.
You can set the speed with the arrow keys and see how the pendulum moves.
"""

import numpy as np
import time
import pymunk               
import pygame
import json
#TODO:??? unused?
#import sys

#initialization of the environment
pygame.init()

display = pygame.display.set_mode((1200,600))

clock = pygame.time.Clock()
space = pymunk.Space()
FPS = 50

xcamera = 0
ycamera = 300

class Wind():
    
    def __init__(self,base_force=300,force_variance=200,constancy=1,changeability=0.01):
        self.base_force=base_force
        self.force_variance=force_variance
        self.constancy=constancy
        self.wind=0
        self.changeability=changeability

    def blow(self):
        if(np.random.random()<self.changeability):
            self.wind=np.sign(np.random.random()*2-1)*(self.base_force+np.random.randint(0,self.force_variance))
    
class Box():
    '''
    This class rappresent physical object, with a predetermined inizial position,size, density and color

    Methods:
    --------
    MoveX(offset):
        Moves the box of a given offset on the X axis.
    draw():
        Draws the box on the pygame window
    '''
    def __init__ (self, x ,y,width,height,density = 1,  static=False, color =(0,0,0)):
        '''
        Create an istance of box.

        Parameters:
        ----------
        x,y: float
            initial position.
        width: int
            the width of the object. It must be positive.
        height: int
            the height of the object. It must be positive
        density (optional): int
            the density of the object. The default value is 1.
        static (optional): boolean
            decide if the body is code-driven(True) or physics-driven(False). The default value is False.
        color (optional): tuple of size 3
            the color of the object, rappresented in RGB notation. The default value is (0,0,0).

        Returns:
            Box
                a box instance with the given values.
        '''
        #setup the attributes with the parameters
        if static:
            self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        else:
            self.body = pymunk.Body(5, 500)
        self.body.position = x, y
        self.width = width
        self.height = height
        self.shape = pymunk.Poly.create_box(self.body, (width, height))
        self.shape.density = density
        space.add(self.body, self.shape)
        self.color = color

    def moveX(self,offset):
        '''
        Moves the box of a given offset on the X axis.

        Parameters:
        ----------
        offset: int
            the given offset.
        '''
        #self.body.position = self.body.position[0] + offset, self.body.position[1]
        self.body.velocity = offset, 0
    def draw(self):
        '''
        Draws the box on the pygame window
        '''
        x, y = self.body.position
        pygame.draw.rect(display, self.color,(int(x-self.width/2) - xcamera, int(y- self.height/2) , self.width, self.height))

class String():
    '''
    This class rappresent a black physical string that connect two bodies.
    
    Methods:
    --------
    draw():
        Draws the box on the pygame window
    '''
    def __init__(self, body1, attachment):
        '''
        Create an instance of String that connect two bodies.

        Parameters:
        ----------
        body1: body
            the first body to connect.
        attachment: body
            the second body to connect.
        
        Returns:
        --------
        String
            the string that connect the two bodies.
        '''
        self.body1 = body1
        self.body2 = attachment
        self.shape= pymunk.PinJoint(self.body1, self.body2)
        space.add(self.shape)

    def draw(self):
        '''
        Draws the box on the pygame window
        '''
        x1, y1 = self.body1.position
        x2, y2 = self.body2.position
        pygame.draw.line(display,(0,0,0), (int(x1)-xcamera, int(y1)), (int(x2)-xcamera, int(y2)), 2)

class PendulumEnv:
    '''
    This class rappresents the environment of a Pendulum, with an agent
    that can be trained to keep the Pendulum in balance.
    Methods():
    ----------
        get_episilon(alpha):
            Given an alpha, gives the epsilon.
        get_reward():
            Calculate the reward based on the current state.
        UP_or_DOWN():
            Returns the direction of the box.
        get_angle():
            Returns the current angle between the box and the base.
        get_new_state():
            Returns the new state based on current status.
        get_continuos_velocity(velocity):
            Returns the vertical and horizontal speed.
        get_discrete_velocity(velocity):
            Discretize the continuos velolcity.
        episode_status():
            Gives the actual status of the current episode.
        step(action):
            Compute a step, or frame, with the given action.
        sample_cond(i):
            Returns true if i is the last episode.
        train():
            Train the model.
        simulate():
            Simulate the model.
        save_q_table(file):
            Saves the actual qTable in a file.
        load_q_table(file,shape):
            Load the qTable from a save file.
        render():
            Render the environment in his current state.
    '''
    def __init__(self, LEARNING_RATE, DISCOUNT, MAX_EPSILON, MIN_EPSILON, DECAY_RATE, Q_TABLE_DIM,EPISODES, START_BASE, START_BOX,space,Q_TABLE_FILE, TICK_LIMIT = 800, is_train = False):
        '''
        Create an instance of PendulumEnv.

        Parameters:
        -----------
        LEARNING_RATE: float
            the learning rate of the model
        DISCOUNT: float
            the discount factor of the model. 
            Higher the value, higher the importance of the future rewards.
            Lower the value, higher the importance of the actual reward.
        MAX_EPSILON: float
            max value of epsilon
        MIN_EPSILON: float
            min value of the epsilon.
        DECAY_RATE: ???
            TODO:??? Non ho capito a che serve, non lo usiamo.
        Q_TABLE_DIM: tuple
            the shape of the qTable.
        EPISODES: int
            the number of episodes of the training session.
        base: Base
            the base of the pendolum.
        box: Box
            the box of the pendolum.
        string: String
            the string that connect che base and the box.
        space: ???
            the space of pyGame.
        Q_TABLE_FILE(optional): string
            the path where load or save the QTable.
            If empty or invalid, it will create a new QTable and will save as unknow.json
        
        '''
        self.START_BASE = START_BASE
        self.START_BOX = START_BOX
        self.LEARNING_RATE = LEARNING_RATE
        self.DISCOUNT = DISCOUNT
        self.MAX_EPSILON = MAX_EPSILON
        self.MIN_EPSILON = MIN_EPSILON
        self.DECAY_RATE = DECAY_RATE
        self.EPISODES = EPISODES
        self.ANGLE_SAMPLES,self.SPEED_SAMPLES, _,self.ACTION_NUM= Q_TABLE_DIM
        print(self.ANGLE_SAMPLES,self.SPEED_SAMPLES, self.ACTION_NUM)

        self.Q_TABLE_FILE = Q_TABLE_FILE
        self.q_table = np.zeros(Q_TABLE_DIM)
        print("[INFO]\t File name set as: ",self.Q_TABLE_FILE)
            
        self.prev_pos = [0,0]
        self.timer = 0
        self.TICK_LIMIT = TICK_LIMIT
        self.frame_count = 0
        self.space = space
        self.space.gravity = (0, 1000)
        self.action = 0
        self.wind=Wind(base_force=100,force_variance=500,changeability=0.008)
        self.tick=0
        self.is_train = is_train
        self.set_reward_param()

    def get_epsilon(self,alpha):
        '''
        Returns the epsilon, or the "randomness" based on the given alpha and
        the elapsed episodes.

        PARAMETERS
        ----------
        alpha: float

        '''
        r = max((self.EPISODES- alpha)/self.EPISODES, 0)
        return (self.MAX_EPSILON - self.MIN_EPSILON)*r+self.MIN_EPSILON

    def set_reward_param(self, alpha = 0.8, beta = 0.2):
        self.alpha = alpha
        self.beta = beta

    def get_reward(self):
        '''
        Returns the reward based on the current state.
        The reward depends on the angle and on the velocity of the box.
        Less the angle and velocity, highest the reward.

        Returns
        -------
        float
            The reward
        '''
        angle = self.get_angle()
        return self.alpha* np.sin(np.deg2rad(angle)) + self.beta* (50-self.get_discrete_velocity(self.get_continuos_velocity(self.box.body.velocity)))/50 *np.sin(np.deg2rad(angle))
        #return self.alpha* np.sin(np.deg2rad(angle)) - np.sin(np.deg2rad(angle))*self.beta* (self.get_discrete_velocity(self.get_continuos_velocity(self.box.body.velocity)))/(self.SPEED_SAMPLES-1)
        return self.alpha* np.sin(np.deg2rad(angle)) + self.beta* ((self.SPEED_SAMPLES-1)-self.get_discrete_velocity(self.get_continuos_velocity(self.box.body.velocity)))/(self.SPEED_SAMPLES-1)

    def UP_or_DOWN(self):
        '''
        Returns the direction of the box.

        Returns
        -------
        int
            1 is it's going up, 0 otherwise
        '''
        if self.box.body.position[1] > self.prev_pos[1]:
            return 0 #DOWN
        return 1 #UP
        
    def get_angle(self):
        '''
        Get the current angle between the base and the box.

        Returns:
        -------
        int
            the current angle
        '''

        xbox, ybox = self.box.body.position
        xbase, ybase = self.base.body.position
        #TODO:??? Tra verticale sotto e verticale sopra non dovrebbero esserci 180 gradi di differenza?
        if xbase == xbox:
            if ybox-ybase > 0:
                return 270
            else:
                return 90
        #TODO:??? Perchè negare la divisione se poi fai abs?
        #Get the slope of the string.
        mr = np.abs(-(ybox - ybase)/(xbox - xbase))

        def from_0_to_360(x, y, xo, yo, angle):
            '''
            TODO:???? Cosa rappresenta angle?
            '''
            if x-xo > 0 and y-yo < 0:
                return angle
            if x-xo < 0 and y-yo < 0:
                return 180-angle
            if x-xo < 0 and y-yo > 0:
                return 180 + angle
            if x-xo > 0 and y-yo > 0:
                return 360 - angle
        return from_0_to_360(xbox, ybox, xbase, ybase, np.degrees(np.arctan(mr)))

    def get_continuos_velocity(self, velocity):
        '''
        Given the velocity as two vectors, one for vertical speed and one
        for horizontal speed, unify them in a single speed.

        Parameters
        ----------
        velocity: tuple with 2 elements
            the speed expressed as vertical and horizontal.

        Returns
        ------
        float
            the unified speed

        '''
        v1, v2 = velocity
        return np.sqrt(v1*v1 + v2*v2)
        
    def get_discrete_velocity(self, velocity):
        '''
        Given a continuos velocity, discretize it

        Parameters:
        -----------
        velocity:  
            the speed in floating number rappresentation

        Returns:
        --------
        int
            A discretized velocity
        '''
        MAX_VEL = 1420
        discrete_v = min(velocity, MAX_VEL)
        if discrete_v <= 10:
            return int(discrete_v)
        elif discrete_v <= 40:
            return int(11 +  (discrete_v-10)//5)
        elif discrete_v <= 100:
            return int(17 + (discrete_v-40)//10)
        else:
            return int(23 + (discrete_v-100)//20)


    def step(self,action,wind=0):
        '''
        Compute a step, or frame, with the given action taken.
        
        Parameters
        ----------
        action: int
            the code of the given action.

        Returns
        -------
        tuple
            a tuple with this format: (reward,new state,done,truncated)
        '''
        self.prev_pos = self.box.body.position
        self.action = action
        self.base.moveX(action)
        self.box.body.apply_impulse_at_local_point([wind,0],(0,0)) 
        #TODO:???
        self.space.step(1/FPS)
        return self.get_reward(), self.get_new_state(), self.episode_status()[0],self.episode_status()[1]
 
    def sample_cond(self, i):
        '''
        Returns true if i is the last episode

        Parameter
        ---------
        i: int
            the current episode

        Return
        ------
        boolean
            if i is the last episodes
        '''
        return i == (self.EPISODES -1)

    def simulate(self):
        '''
        Starts the simulation.
        '''
        theta= np.random.random()*np.pi*2
        
        
        xOff,yOff= np.cos(theta)*200,np.sin(theta)*200
        print(theta,xOff,yOff)
        self.base= Box(self.START_BASE[0],self.START_BASE[1], 100, 10, static=True)
        self.box = Box(self.START_BASE[0]+xOff,self.START_BASE[1]+yOff, 50, 50, color=(191, 64, 191))
        self.string = String(self.base.body, self.box.body)
        while True:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                       
                        self.base.moveX(50)
                    if event.key == pygame.K_RIGHT:
                        self.base.moveX(250)
                    if(event.key == pygame.K_UP):
                        self.base.moveX(500)
                    if(event.key == pygame.K_DOWN):
                        self.base.moveX(1120)
            
            self.prev_pos = self.box.body.position
            #TODO:???
            self.space.step(1/FPS)
            self.render()
        """
        self.q_table = self.load_q_table(self.Q_TABLE_FILE,self.q_table.shape)
        state = (int(self.get_angle()//(360/self.ANGLE_SAMPLES)),0,0)
        truncated = False

        self.string = String(self.base.body, self.box.body)

        self.timer = time.time()
        while  not truncated:
            action = np.argmax(self.q_table[state[0],state[1],state[2]])
            speed = action%(self.ACTION_NUM//2)*17
            if action > (self.ACTION_NUM//2):
                speed = -speed
            self.wind.blow()
            _,new_state, _, truncated = self.step(speed)
            self.render()
            state = new_state
    """

    def execEnv(self):
            self.simulate()
    def render(self):
        '''
        Render the environment in his current state.
        '''
        global xcamera
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
        txt = pygame.font.SysFont("Arial", 15).render(str(time.time()- self.timer)[:3], True, (0,0,0))
        
        display.fill((255,255,255))
        display.blit(txt, (30, 15))
    
        txt = pygame.font.SysFont("Arial", 15).render("Wind: "+str(self.wind.wind)[:7], True, (0,0,0))
        display.blit(txt, (30, 30))

        txt = pygame.font.SysFont("Arial", 15).render("Reward: "+str(self.get_reward())[:5], True, (0,0,0))
        display.blit(txt, (30, 45))

        txt = pygame.font.SysFont("Arial", 15).render("Cube speed: "+str(self.get_discrete_velocity(self.get_continuos_velocity(self.box.body.velocity))), True, (0,0,0))
        display.blit(txt, (30, 60))

        self.base.draw()
        self.box.draw()
        self.string.draw()
        if self.base.body.position[0] -(xcamera+600)> 600:
            xcamera += 1200
        if (xcamera+600)- self.base.body.position[0]> 600:
            xcamera -= 1200
        pygame.display.update()
        clock.tick(FPS)
"""
Instruction for use:
    - To execute for training set is_train to True in the initialization of the environment, False for simulate;
    - In any case is necessary specify the file name on which save the table (variable Q_TABLE_FILE);
    - To set reward parameters (alpha, beta) use the set_reward_param function 
"""


if __name__ == "__main__":
    Q_TABLE_FILE ="test.json"
    env = PendulumEnv(LEARNING_RATE = 0.5, DISCOUNT=0.98, MAX_EPSILON=1.0, MIN_EPSILON=0.05, DECAY_RATE=0.005, 
                      Q_TABLE_DIM = (40, 90, 2, 80),EPISODES=5000000,START_BOX=(600, 500), START_BASE=(600, 300),
                      space=space,Q_TABLE_FILE=Q_TABLE_FILE, is_train=False)
    env.set_reward_param(0.6, 0.4)
    pygame.display.set_caption(Q_TABLE_FILE)
    env.execEnv()
    