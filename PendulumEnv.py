
import numpy as np
import time
import pymunk               
import pygame
import json
from Graphics import Box, Wind, String
'''
Initialization of pygame and pymunk environment
'''
pygame.init()

display = pygame.display.set_mode((1200,600))

clock = pygame.time.Clock()
space = pymunk.Space()
FPS = 50

xcamera = 0
ycamera = 300

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
            Load the qTable from a saved file.
        render():
            Render the environment in his current state.
        set_reward_param(alpha,beta):
            Set the weight of each component of the reward function.
        write_on_log():
            Write on log.txt the parameters of the current training session
        
        

    '''
    def __init__(self, LEARNING_RATE, DISCOUNT, MAX_EPSILON, MIN_EPSILON, Q_TABLE_DIM,EPISODES, START_BASE, START_BOX,space,Q_TABLE_FILE, TICK_LIMIT = 800, is_train = False):
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

        Q_TABLE_DIM: tuple
            the shape of the qTable.
        EPISODES: int
            the number of episodes of the training session.
        START_BASE:
            the initial position of the base.
        START_BOX:
            the initial position of the cube.
        space: pymunk.Space
            the space of pyGame.
        Q_TABLE_FILE(optional): string
            the path where load or save the QTable.
            If empty or invalid, it will create a new QTable and will save as unknow.json
        TICK_LIMIT: int
            the maximum number of iteration of one training episode.
        is_train: bool
            if True a trainig session will start otherwise simulate the table in Q_TABLE_FILE.
        
        '''
        self.START_BASE = START_BASE
        self.START_BOX = START_BOX
        self.LEARNING_RATE = LEARNING_RATE
        self.DISCOUNT = DISCOUNT
        self.MAX_EPSILON = MAX_EPSILON
        self.MIN_EPSILON = MIN_EPSILON
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
        self.space.damping=0.9
        self.action = 0
        self.wind=Wind(base_force=0,force_variance=300,changeability=0.008)
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

        RETURN
        ------
        new_epsilon: float
        '''
        r = max((self.EPISODES- alpha)/self.EPISODES, 0)
        return (self.MAX_EPSILON - self.MIN_EPSILON)*r+self.MIN_EPSILON

    def set_reward_param(self, alpha = 0.8, beta = 0.2):
        '''
        Set the weight of each component of the reward function.
        
        PARAMETERS
        ----------
        alpha: float
        beta: float
        '''
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
        return self.alpha* np.sin(np.deg2rad(angle)) + self.beta* (20-self.get_discrete_velocity(self.get_continuos_velocity(self.box.body.velocity)))/20
        
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
            the current angle in degrees
        '''

        xbox, ybox = self.box.body.position
        xbase, ybase = self.base.body.position

        if xbase == xbox:
            if ybox-ybase > 0:
                return 270
            else:
                return 90

        #Get the slope of the string.
        mr = np.abs(-(ybox - ybase)/(xbox - xbase))

        def from_0_to_360(x, y, xo, yo, angle):
            '''
            Convert the input angle to an angle between 0 and 360

            PARAMETERS
            ----------
            x: int
                x-coordinate of the first object.
            y: int
                y-coordinate of the first object.
            xo: int
                x-coordinate of the second object.
            yo: int
                y-coordinate of the second object.
            angle: int
                angle between 0-90 degrees.

            RETURN
            ------
            int
                angle between 0 and 360
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

    def get_new_state(self):
        '''
        Given the current angle,velocity and direction, gives a new state.

        Returns
        -------
        tuple
            The new state in the form of (angle,velocity,direction)
        '''
        angle = self.get_angle()
        return int(angle//(360/self.ANGLE_SAMPLES)), self.get_discrete_velocity(self.get_continuos_velocity(self.box.body.velocity)), self.UP_or_DOWN() 

    def get_continuos_velocity(self, velocity):
        '''
        Given the velocity as a vectors of two coordinates, one for vertical speed and one
        for horizontal speed, unify them in a single speed.

        Parameters
        ----------
        velocity: tuple with 2 elements
            the speed expressed as vertical and horizontal.

        Returns
        ------
        float
            the unified speed.

        '''
        v1, v2 = velocity
        return np.sqrt(v1*v1 + v2*v2)
        
    def get_discrete_velocity(self, velocity):
        '''
        Given a continuos velocity, discretize it in 20 buckets.
        The dimension of the backet increases as the value of the velocity gets higher.

        Parameters:
        -----------
        velocity:  
            the speed in floating number rappresentation.

        Returns:
        --------
        int
            A discretized velocity.
        '''
        MAX_VEL = 1670
        #TOT 19
        discrete_v = min(velocity, MAX_VEL)
        if  discrete_v <= 120:
            return int((discrete_v)//30)
        elif discrete_v <= 420:
            return int(4 + (discrete_v-120)//60)
        elif discrete_v <= 920:
            return int(9 + (discrete_v-420)//100)
        else:
            return int(14 + (discrete_v-920)//150)

    def episode_status(self):
        '''
        Returns the actual status of the current episode. 
        If the box has been in the right spot for 5 seconds, 
        then end successfully the episode.
        After TICK_LIMIT iterations the episode is truncated.

        Returns
        -------
        tuple of two elements
            A tuple of boolean that contains if the episode is ended
            or truncated.
        '''
        state = self.get_new_state()
        #if the box is the right spot (it's vertical)
        if state[0] >= (89//(360/self.ANGLE_SAMPLES)) and state[0] <= (91//(360/self.ANGLE_SAMPLES)):
            self.box.color = (0,255,0)
            self.frame_count+=1
            if( self.frame_count > 5*FPS):
                return (True, False)
        #box fallen. Truncate
        elif state[0] >= (269//(360/self.ANGLE_SAMPLES)) and state[0] <= (271//(360/self.ANGLE_SAMPLES)): 
            self.box.color = (255, 0,0)
            self.timer = time.time()
            self.frame_count=0
            return False,(self.tick > self.TICK_LIMIT)
        else:
            self.timer = time.time()
            self.frame_count=0
            self.box.color = (191, 64, 191)
        return (False, False) 

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

    def write_on_log(self):
        '''
        Write on log.txt the parameters of the current training session
        '''
        with open('log.txt', 'a') as log:
            record = "<Q_TABLE: "+str(self.Q_TABLE_FILE) +", ANGLE_SAMPLES: "+str(self.ANGLE_SAMPLES)+", SPEED_SAMPLES: "+str(self.SPEED_SAMPLES)+", ACTION_NUM: "+str(self.ACTION_NUM)+", EPISODES: "\
                +str(self.EPISODES) +", START_BASE: "+ str(self.START_BASE) +", START_BOX: "+ str(self.START_BOX) +", LEARNING_RATE: "+ str(self.LEARNING_RATE) +\
                ", DISCOUNT: "+ str(self.DISCOUNT) + ", MAX_EPSILON: "+str(self.MAX_EPSILON)+ ", MIN_EPSILON: "+str(self.MIN_EPSILON)\
                    +", REWARD_ALPHA: " + str(self.alpha) + ", REWARD_BETA: "+ str(self.beta)+">\n"
            log.write(record)

    def train(self):
        '''
        Train the model apllying the Q_Learning algorithm. 
        '''
        self.write_on_log()

        global xcamera
        global ycamera
        cmd_t = 0
        successes = 0
        input("\nPress any key to start\n")
        for episode in range(self.EPISODES):
            theta= np.random.random()*np.pi*2
        
            '''
            Initialize the training environment.
            '''
            xOff,yOff= np.cos(theta)*200,np.sin(theta)*200
            self.base= Box(self.START_BASE[0],self.START_BASE[1], 100, 10, static=True, space=space)
            self.box = Box(self.START_BASE[0]+xOff,self.START_BASE[1]+yOff, 50, 50, color=(191, 64, 191), space=space)
            self.string = String(self.base.body, self.box.body, space=space)
            self.tick = 0
            
            done = False
            render = False
            truncated = False
            xcamera = 0
            ycamera = 300
            
            '''
            The state has the format: (angle,velocity,direction).
            '''
            state = (int(self.get_angle()//(360/self.ANGLE_SAMPLES)),0,1)
            
            '''
            Get the current epsilon.
            '''
            epsilon = self.get_epsilon(episode)**(1.7)
            actualLR=self.LEARNING_RATE
            if self.sample_cond(episode):
                input("Last episode")
            line = ''

            '''
            Read cmd.txt to check if a command were given.
            Command:
                - save: save the actual q_table on Q_TABLE_FILE;
                - exit: end the session;
                - show: render the training scene;
                - status: print some paramenters (current episode, success rate, actual lr, actual epsilon);
            '''
            with open('cmd.txt', 'r')as cmd:
                line = cmd.readline()
                if line: 
                    if line == 'save' and not(cmd_t == 1):
                        self.save_q_table(self.Q_TABLE_FILE)
                        input("Saved. Press enter to continue")
                        cmd_t = 1
                    if line == 'exit':
                        cmd_t= 2
                        break
                    if line == 'show':
                        cmd_t = 3
                        render = True
                    if line == 'status':
                        cmd_t = 4
                        print("EPISODE: ", episode)
                        print("SUCCESS RATE: ", successes/(episode+1))
                        print("ACTUAL LR: ", actualLR)
                        print("ACTUAL EPSILON: ", epsilon)
 
                else:
                    cmd_t = 0
                    
            self.timer = time.time()

            '''
            Training Loop
            '''
            while not done and not truncated:
                #check if has to do the best move given the actual table or a random move.
                if np.random.random() > epsilon:
                    action = np.argmax(self.q_table[state[0],state[1],state[2]])
                else:
                    
                    action = np.random.randint(0, self.ACTION_NUM)

                #convert the speed from the coded state to the actual uncoded speed
                speed = (action%(self.ACTION_NUM//2)) * 50

                #the first half are positive values, the second one are negative
                if action > (self.ACTION_NUM//2):
                    speed = -speed

                #make a step
                reward,new_state, done, truncated = self.step(speed)

                #training ended or render requested
                if self.sample_cond(episode) or render:
                    self.render()

                #Q_Learning algorithm
                max_future_q = np.max(self.q_table[new_state[0],new_state[1],new_state[2]])
                current_q = self.q_table[state[0]][state[1]][state[2]][action]

                new_q = (1-actualLR) * current_q + actualLR * (reward + self.DISCOUNT * max_future_q)
                
                self.q_table[state[0],state[1],state[2]][action] = new_q
                state = new_state
                self.tick+=1

               

            if done:
                successes += 1
            
            
            #remove the last episode objects, if present.
            space.remove(self.base.shape, self.base.body)
            space.remove(self.box.shape, self.box.body)
            space.remove(self.string.shape)
            

        self.save_q_table(self.Q_TABLE_FILE)

    def save_q_table(self, file):
        '''
        Saves the actual qTable in a file.

        Parameters
        ----------
        file:
            the path of the save file.
        '''
        with open(file,'w') as f:
            tosave =[]
            x,y,z,d = self.q_table.shape
            for i in range(x):
                for j in range(y):
                    for q in range(z):
                        for w in range(d):
                            tosave.append(self.q_table[i][j][q][w])
            json.dump(tosave, f)
        print('Q_TABLE SAVED ON '+ self.Q_TABLE_FILE)

    def load_q_table(self, file, shape):
        '''
        Load the qTable from a save file

        Parameters
        ----------
        file:
            the path of the save file.
        shape:
            the expected shape of the table in the save file.
        '''
        q_table = np.zeros(shape)
        
        try:
            q_list =[]
            f=open(file,"r")
            q_list = json.load(f)
            ind = 0
            x,y,z,d = shape
            for i in range(x):
                for j in range(y):
                    for q in range(z):
                        for w in range(d):
                            q_table[i][j][q][w] = q_list[ind]
                            ind += 1
            f.close()
            print("[INFO]\t File loaded with success")
        except FileNotFoundError:
            print("[ERROR]\t File not found, using an empty table")

        return q_table
    def simulate(self):
        '''
        Starts the simulation.
        '''
        '''
        Initialize the training environment.
        The box is initialized with a random angle.
        '''
        theta= np.random.random()*np.pi*2 
        
        xOff,yOff= np.cos(theta)*200,np.sin(theta)*200
        self.base= Box(self.START_BASE[0],self.START_BASE[1], 100, 10, static=True, space=space)
        self.box = Box(self.START_BASE[0]+xOff,self.START_BASE[1]+yOff, 50, 50, color=(191, 64, 191),space=space)
        self.string = String(self.base.body, self.box.body,space=space)

        self.q_table = self.load_q_table(self.Q_TABLE_FILE,self.q_table.shape)
        state = (int(self.get_angle()//(360/self.ANGLE_SAMPLES)),0,1)

        print("[INFO]\t Starting at angle:",self.get_angle())
        truncated = False

        self.timer = time.time()
        '''
        Simulation loop
        '''
        while not truncated:
            action = np.argmax(self.q_table[state[0],state[1],state[2]])
            speed = action%(self.ACTION_NUM//2)*50
            if action > (self.ACTION_NUM//2):
                speed = -speed
            if state[0] >= (89//(360/self.ANGLE_SAMPLES)) and state[0] <= (91//(360/self.ANGLE_SAMPLES)):
                self.wind.blow()
                self.box.body.apply_impulse_at_local_point([self.wind.wind,0],(0,0)) 
            _,new_state, _, truncated = self.step(speed)
            self.render()
            state = new_state
            
    def execEnv(self):
        '''
        Based on the attribute is_train, starts the training or the simulation.
        '''
        if self.is_train:
            self.train()
        else:
            self.simulate()

    def render(self):
        '''
        Render the environment in his current state and print some parameters.
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

        self.base.draw(display, xcamera)
        self.box.draw(display, xcamera)
        self.string.draw(display, xcamera)
        if self.base.body.position[0] -(xcamera+600)> 600:
            xcamera += 1200
        if (xcamera+600)- self.base.body.position[0]> 600:
            xcamera -= 1200
        pygame.display.update()
        clock.tick(FPS)

