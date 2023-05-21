
import numpy as np
import time
import pymunk               
import pygame
import json


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
        self.body.velocity = (offset, 0)

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
        self.wind=Wind(base_force=0,force_variance=1,changeability=0.008)
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

    def get_learning_rate(self,alpha):

        r = max((self.EPISODES- alpha)/self.EPISODES, 0)
        return (self.LEARNING_RATE - 0.05)*r+0.05
    
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
        return self.alpha* np.sin(np.deg2rad(angle)) + self.beta* (20-self.get_discrete_velocity(self.get_continuos_velocity(self.box.body.velocity)))/20
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
        If the box has been in the right spot for 10 second, 
        then end successfully the episode.
        If the box is on the bottom (fallen), then truncate the episode.
        Keep going otherwise.

        Returns
        -------
        tuple of two elements
            A tuple of boolean that contains if the episode is ended(on position 0)
            or truncated (on position 1).
        '''
        state = self.get_new_state()
        #if the box is the right spot (it's vertical)
        if state[0] >= (89//(360/self.ANGLE_SAMPLES)) and state[0] <= (91//(360/self.ANGLE_SAMPLES)):
            self.box.color = (0,255,0)
            self.frame_count+=1
            #print(self.frame_count,10*FPS,FPS)     
            #if (time.time()-self.timer) > 10:
            #second to frame conversion
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
    def write_on_log(self):
        with open('log.txt', 'a') as log:
            record = "<Q_TABLE: "+str(self.Q_TABLE_FILE) +", ANGLE_SAMPLES: "+str(self.ANGLE_SAMPLES)+", SPEED_SAMPLES: "+str(self.SPEED_SAMPLES)+", ACTION_NUM: "+str(self.ACTION_NUM)+", EPISODES: "\
                +str(self.EPISODES) +", START_BASE: "+ str(self.START_BASE) +", START_BOX: "+ str(self.START_BOX) +", LEARNING_RATE: "+ str(self.LEARNING_RATE) +\
                ", DISCOUNT: "+ str(self.DISCOUNT) + ", MAX_EPSILON: "+str(self.MAX_EPSILON)+ ", MIN_EPSILON: "+str(self.MIN_EPSILON)\
                    +", REWARD_ALPHA: " + str(self.alpha) + ", REWARD_BETA: "+ str(self.beta)+">\n"
            log.write(record)
    def train(self):
        '''
        Train the model.
        '''
        self.write_on_log()

        global xcamera
        global ycamera
        cmd_t = 0
        successes = 0
        input("\nPress any key to start\n")
        for episode in range(self.EPISODES):
            theta= np.random.random()*np.pi*2
        
        
            xOff,yOff= np.cos(theta)*200,np.sin(theta)*200
            self.base= Box(self.START_BASE[0],self.START_BASE[1], 100, 10, static=True)
            self.box = Box(self.START_BASE[0]+xOff,self.START_BASE[1]+yOff, 50, 50, color=(191, 64, 191))
            self.string = String(self.base.body, self.box.body)
            self.tick = 0
            
            done = False
            render = False
            truncated = False
            xcamera = 0
            ycamera = 300
            
            '''
            The state has the format: (angle,velocity,direction)
            '''
            state = (int(self.get_angle()//(360/self.ANGLE_SAMPLES)),0,1)
            
            epsilon = self.get_epsilon(episode)**(1.7)
            #actualLR = self.get_learning_rate(episode)
            actualLR=self.LEARNING_RATE#1/(1+episode)**self.LEARNING_RATE
            if self.sample_cond(episode):
                input("Last episode")
            line = ''

            #check commands from cmd.txt
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

            

            #training loop
            while not done and not truncated:
                #check if has to do the best move given the actual table or a random move.
                if np.random.random() > epsilon:
                    #TODO:???? Perchè argmax?
                    action = np.argmax(self.q_table[state[0],state[1],state[2]])
                else:
                    
                    action = np.random.randint(0, self.ACTION_NUM)
                #convert the speed from the coded state to the actual uncoded speed
                speed = (action%(self.ACTION_NUM//2)) * 50
                
                if action > (self.ACTION_NUM//2):
                    speed = -speed

                #make a step
                reward,new_state, done, truncated = self.step(speed)

                #training ended or render requested
                if self.sample_cond(episode) or render:
                    self.render()

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

        theta= np.random.random()*np.pi*2
        
        
        xOff,yOff= np.cos(theta)*200,np.sin(theta)*200
        self.base= Box(self.START_BASE[0],self.START_BASE[1], 100, 10, static=True)
        self.box = Box(self.START_BASE[0],self.START_BASE[1]+200, 50, 50, color=(191, 64, 191))
        self.string = String(self.base.body, self.box.body)

        self.q_table = self.load_q_table(self.Q_TABLE_FILE,self.q_table.shape)
        state = (int(self.get_angle()//(360/self.ANGLE_SAMPLES)),0,1)
        truncated = False

        self.string = String(self.base.body, self.box.body)

        self.timer = time.time()
        while  not truncated:
            action = np.argmax(self.q_table[state[0],state[1],state[2]])
            speed = action%(self.ACTION_NUM//2)*50
            if action > (self.ACTION_NUM//2):
                speed = -speed
            self.wind.blow()
            _,new_state, _, truncated = self.step(speed)
            self.render()
            state = new_state
    def execEnv(self):
        if self.is_train:
            self.train()
        else:
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
    env = PendulumEnv(LEARNING_RATE = 0.1, DISCOUNT=0.95, MAX_EPSILON=1.0, MIN_EPSILON=0.05, 
                      Q_TABLE_DIM = (40, 20, 2, 20),EPISODES=20000,START_BOX=(600, 500), START_BASE=(600, 300),
                      space=space,Q_TABLE_FILE=Q_TABLE_FILE, is_train=True)
    env.set_reward_param(0.5, 0.5)
    pygame.display.set_caption(Q_TABLE_FILE)
    env.execEnv()
    