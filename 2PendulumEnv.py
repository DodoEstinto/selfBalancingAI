#TODO:??? unused?
#import cv2
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
    def __init__(self, LEARNING_RATE, DISCOUNT, MAX_EPSILON, MIN_EPSILON, DECAY_RATE, Q_TABLE_DIM,EPISODES, START_BASE, START_BOX1,START_BOX2,space,Q_TABLE_FILE, TICK_LIMIT = 800, is_train = False):
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
        self.START_BOX1 = START_BOX1
        self.START_BOX2 = START_BOX2
        self.LEARNING_RATE = LEARNING_RATE
        self.DISCOUNT = DISCOUNT
        self.MAX_EPSILON = MAX_EPSILON
        self.MIN_EPSILON = MIN_EPSILON
        self.DECAY_RATE = DECAY_RATE
        self.EPISODES = EPISODES
        self.ANGLE_SAMPLES,self.SPEED_SAMPLES, _,_,_,_,self.ACTION_NUM= Q_TABLE_DIM
        self.TICK_LIMIT = TICK_LIMIT
        print(self.ANGLE_SAMPLES,self.SPEED_SAMPLES, self.ACTION_NUM)

        self.Q_TABLE_FILE = Q_TABLE_FILE
        self.q_table = np.zeros(Q_TABLE_DIM)
        print("[INFO]\t File name set as: ",self.Q_TABLE_FILE)
            
        self.prev_pos1 = START_BOX1
        self.prev_pos2 = START_BOX2
        self.timer = 0
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
        #if the box is the right spot (it's vertical)
        x = self.box1.body.position[0]
        y = 600 -self.box2.body.position[1]
        angle1 = self.get_angle(self.base.body.position,self.box1.body.position)
        angle2 = self.get_angle(self.box1.body.position,self.box2.body.position)
        dir = (self.UP_or_DOWN(self.box2.body.position,self.prev_pos2))
        if dir == 0:
            dir = -1
        """
        alive = 0
        if angle2 >= 87 and angle2 <= 93:
            alive = 10
        return alive -0.01 * x*x + (y - 2)*(y-2)
        return self.UP_or_DOWN(self.box2.body.position, self.prev_pos2) -1"""
        return dir*np.exp((dir*-5* np.sqrt( self.alpha*np.power(np.cos(np.deg2rad(angle1)),(2) )+self.beta* np.power(np.cos(np.deg2rad(angle2)), (2)) ) ) )
        dir  = self.UP_or_DOWN(self.box2.body.position,self.prev_pos2)
        if dir == 0:
            dir = -1
        state = self.get_new_state()
        if state[0] >= (181//(360/self.ANGLE_SAMPLES)) or state[3] >= (181//(360/self.ANGLE_SAMPLES)):
            return -100
        return - np.abs(self.beta*np.cos(np.deg2rad(angle2))) - np.abs(self.alpha*np.cos(np.deg2rad(angle1)))
        #self.beta* (np.sin(np.deg2rad(angle1)) )+dir*self.alpha* np.abs(np.sin(np.deg2rad(angle2)))
        #return self.alpha* np.sin(np.deg2rad(angle1)) +self.alpha* np.sin(np.deg2rad(angle2)) + self.beta* (50-self.get_discrete_velocity(self.get_continuos_velocity(self.box.body.velocity)))/50
        #return self.alpha* np.sin(np.deg2rad(angle)) - np.sin(np.deg2rad(angle))*self.beta* (self.get_discrete_velocity(self.get_continuos_velocity(self.box.body.velocity)))/(self.SPEED_SAMPLES-1)
        #return self.alpha* np.sin(np.deg2rad(angle)) + self.beta* ((self.SPEED_SAMPLES-1)-self.get_discrete_velocity(self.get_continuos_velocity(self.box.body.velocity)))/(self.SPEED_SAMPLES-1)

    def UP_or_DOWN(self, pos, pre_pos):
        '''
        Returns the direction of the box.

        Returns
        -------
        int
            1 is it's going up, 0 otherwise
        '''
        if pos[1] > pre_pos[1]:
            return 0 #DOWN
        return 1 #UP
        
    def get_angle(self, pos1, pos2):
        '''
        Get the current angle between the base and the box.

        Returns:
        -------
        int
            the current angle
        '''

        xbox, ybox = pos2
        xbase, ybase = pos1
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
        angle1 = self.get_angle(self.base.body.position,self.box1.body.position)
        angle2 = self.get_angle(self.box1.body.position,self.box2.body.position)
        return int(angle1//(360/self.ANGLE_SAMPLES)), self.get_discrete_velocity(self.get_continuos_velocity(self.box1.body.velocity)), self.UP_or_DOWN(self.box1.body.position, self.prev_pos1),\
            int(angle2//(360/self.ANGLE_SAMPLES)),self.get_discrete_velocity(self.get_continuos_velocity(self.box2.body.velocity)), self.UP_or_DOWN(self.box2.body.position, self.prev_pos2) 

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
        MAX_VEL = 700
        discrete_v = min(velocity, MAX_VEL)
        if discrete_v <= 10:
            return int(discrete_v)
        elif discrete_v <=  100:
            return int(11 +  (discrete_v-10)//10)
        elif discrete_v <= MAX_VEL:
            return int(21 + (discrete_v-100)//20)
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
        if state[3] >= (89//(360/self.ANGLE_SAMPLES)) and state[3] <= (91//(360/self.ANGLE_SAMPLES)) and state[0] >= (89//(360/self.ANGLE_SAMPLES)) and state[0] <= (91//(360/self.ANGLE_SAMPLES)):
            self.box1.color = (0,255,0)
            self.frame_count+=1

            #print(self.frame_count,10*FPS,FPS)     
            #if (time.time()-self.timer) > 10:
            #second to frame conversion
            
            if( self.frame_count > FPS):
                return (True, False)
        #box fallen. Truncate
        elif state[0] >= (181//(360/self.ANGLE_SAMPLES)) or state[3] >= (181//(360/self.ANGLE_SAMPLES)):#state[0] >= (269//(360/self.ANGLE_SAMPLES)) and state[0] <= (271//(360/self.ANGLE_SAMPLES)): 
            self.box1.color = (255, 0,0)
            self.box2.color = (255, 0,0)
            self.timer = time.time()
            self.frame_count=0
            return False,True#(self.tick > self.TICK_LIMIT)
        else:
            self.timer = time.time()
            self.frame_count=0
            self.box1.color = (191, 64, 191)
            self.box2.color = (191, 64, 191)
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
        self.prev_pos1 = self.box1.body.position
        self.prev_pos2 = self.box2.body.position
        self.action = action
        self.base.moveX(action)
        #TODO:???
        
        self.space.step(1/FPS)
        episode = self.episode_status()
        return self.get_reward(), self.get_new_state(), episode[0],episode[1]
 
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
                +str(self.EPISODES) +", START_BASE: "+ str(self.START_BASE) +", START_BOX: "+ str(self.START_BOX1) +", LEARNING_RATE: "+ str(self.LEARNING_RATE) +\
                ", DISCOUNT: "+ str(self.DISCOUNT) + ", MAX_EPSILON: "+str(self.MAX_EPSILON)+ ", MIN_EPSILON: "+str(self.MIN_EPSILON) +", DECAY_RATE: "+ str(self.DECAY_RATE)\
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
        max_rew = -100000
        max_state = (0,0,0,0,0,0)
        max_act = -100000
        last_100 = []
        input("\nPress any key to start\n")
        for episode in range(self.EPISODES):
            
            #generate the new objects.
            self.base= Box(self.START_BASE[0],self.START_BASE[1], 100, 10, static=True)
            self.box1 = Box(self.START_BOX1[0],self.START_BOX1[1], 50, 50, color=(191, 64, 191))
            self.box2 = Box(self.START_BOX2[0],self.START_BOX2[1], 50, 50, color=(191, 64, 191))
            self.string1 = String(self.base.body, self.box1.body)
            self.string2 = String(self.box1.body, self.box2.body)
            self.prev_pos1 = self.START_BOX1
            self.prev_pos2 = self.START_BOX2
            self.frame_count = 0
            self.tick = 0
            
            done = False
            render = False
            truncated = False
            xcamera = 0
            ycamera = 300
            
            '''
            The state has the format: (angle,velocity,direction)
            '''

            angle1 = self.get_angle(self.base.body.position,self.box1.body.position)
            angle2 = self.get_angle(self.box1.body.position,self.box2.body.position)
            state = (int(angle1//(360/self.ANGLE_SAMPLES)),0,1, int(angle2//(360/self.ANGLE_SAMPLES)),0,1)
            
            epsilon = self.get_epsilon(episode)

            
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
                        print("SUCCESS RATE: ", np.sum(last_100)/len(last_100))
                else:
                    cmd_t = 0
                    
            self.timer = time.time()
            
            #training loop
            while (not done and not truncated):
                #check if has to do the best move given the actual table or a random move.
                if np.random.random() > epsilon or episode == self.EPISODES -1:
                    #TODO:???? Perchè argmax?
                    action = np.argmax(self.q_table[state[0],state[1],state[2], state[3],state[4],state[5]])
                else:
                    action = np.random.randint(0, self.ACTION_NUM)
                #convert the speed from the coded state to the actual uncoded speed
                speed = (action%(self.ACTION_NUM//2)) * 17
                
                if action > (self.ACTION_NUM//2):
                    speed = -speed

                #make a step
                reward,new_state, done, truncated = self.step(speed)
                #training ended or render requested
                if self.sample_cond(episode) or render:
                    self.render()
                #TODO:??? Arrivato a questo punto credo di sapere concettualmente
                #cosa rappresenta la tabella, ma non come è strutturata bene all'interno.
                #mi manca l'ultima dimensione(quella che nel save prende la variabile "d")
                max_future_q = np.max(self.q_table[new_state[0],new_state[1],new_state[2], new_state[3],new_state[4],new_state[5]])
                current_q = self.q_table[state[0],state[1],state[2], state[3],state[4],state[5]][action]
                lr = self.LEARNING_RATE #*-episode/self.EPISODES*0.01
                new_q = (1-lr) * current_q + lr * (reward + self.DISCOUNT * max_future_q) 
                if reward > max_rew:
                    max_state = state
                    max_rew = reward
                    max_act = action
                
                self.q_table[state[0],state[1],state[2], state[3],state[4],state[5]][action] = new_q
                state = new_state

                self.tick+=1
            if len(last_100) < 100:
                if done: 
                    last_100.append(1)
                else:
                    last_100.append(0)
            else:
                if done: 
                    last_100[episode%100] = 1
                else:
                    last_100[episode%100] = 0
            
            #remove the last episode objects, if present.
            space.remove(self.base.shape, self.base.body)
            space.remove(self.box1.shape, self.box1.body)
            space.remove(self.box2.shape, self.box2.body)
            space.remove(self.string1.shape)
            space.remove(self.string2.shape)
            

        #self.save_q_table(self.Q_TABLE_FILE)

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
            x,y,z,a,b,c,d = self.q_table.shape
            for i in range(x):
                for j in range(y):
                    for q in range(z):
                        for e in range(a):
                            for h in range(b):
                                for g in range(c):
                                    for w in range(d):
                                        tosave.append(self.q_table[i][j][q][e][h][g][w])
            json.dump(tosave, f)
        print('Q_TABLE SAVED.')

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
            x,y,z,a,b,c,d = self.q_table.shape
            for i in range(x):
                for j in range(y):
                    for q in range(z):
                        for e in range(a):
                            for h in range(b):
                                for g in range(c):
                                    for w in range(d):
                                        q_table[i][j][q][e][h][g][w] = q_list[ind]
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
        self.base= Box(self.START_BASE[0],self.START_BASE[1], 100, 10, static=True)
        self.box1 = Box(self.START_BOX1[0],self.START_BOX1[1], 50, 50, color=(191, 64, 191))
        self.box2 = Box(self.START_BOX2[0],self.START_BOX2[1], 50, 50, color=(191, 64, 191))
        self.string1 = String(self.base.body, self.box1.body)
        self.string2 = String(self.box1.body, self.box2.body)

        self.q_table = self.load_q_table(self.Q_TABLE_FILE,self.q_table.shape)
        input("START")

       
        angle1 = self.get_angle(self.base.body.position,self.box1.body.position)
        angle2 = self.get_angle(self.box1.body.position,self.box2.body.position)
        state = (int(angle1//(360/self.ANGLE_SAMPLES)),0,0, int(angle2//(360/self.ANGLE_SAMPLES)),0,0)
        truncated = False


        self.timer = time.time()
        while  not truncated:
            action = np.argmax(self.q_table[state[0],state[1],state[2], state[3],state[4],state[5]])
            speed = action%(self.ACTION_NUM//2)*17
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
        #input("Next frame...")
        global xcamera
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
        txt = pygame.font.SysFont("Arial", 15).render(str(time.time()- self.timer)[:3], True, (0,0,0))
        
        display.fill((255,255,255))
        display.blit(txt, (30, 15))
    
        txt = pygame.font.SysFont("Arial", 15).render("Reward: "+str(self.get_reward()), True, (0,0,0))
        display.blit(txt, (30, 45))
        state = self.get_new_state()
        
        txt = pygame.font.SysFont("Arial", 15).render("Cube1 angle: "+str(state[0]), True, (0,0,0))
        display.blit(txt, (30, 60))

        txt = pygame.font.SysFont("Arial", 15).render("Cube1 speed: "+str(state[1]), True, (0,0,0))
        display.blit(txt, (30, 75))

        txt = pygame.font.SysFont("Arial", 15).render("Direction1: "+str(state[2]), True, (0,0,0))
        display.blit(txt, (30, 90))

        txt = pygame.font.SysFont("Arial", 15).render("Cube2 angle: "+str(state[3]), True, (0,0,0))
        display.blit(txt, (30, 105))

        txt = pygame.font.SysFont("Arial", 15).render("Cube2 speed: "+str(state[4]), True, (0,0,0))
        display.blit(txt, (30, 120))

        txt = pygame.font.SysFont("Arial", 15).render("Direction2: "+str(state[5]), True, (0,0,0))
        display.blit(txt, (30, 135))
        
        txt = pygame.font.SysFont("Arial", 15).render("INitial state: "+str(np.argmax(self.q_table[int(90//(360/self.ANGLE_SAMPLES)),0,1,int(90//(360/self.ANGLE_SAMPLES)),0,1])), True, (0,0,0))
        display.blit(txt, (30, 150))

        

        self.base.draw()
        self.box1.draw()
        self.box2.draw()
        self.string1.draw()
        self.string2.draw()
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
    Q_TABLE_FILE ="2pend.2.json"
    env = PendulumEnv(LEARNING_RATE = 0.7, DISCOUNT=0.996, MAX_EPSILON=1.0, MIN_EPSILON=0.005, DECAY_RATE=0.005, 
                      Q_TABLE_DIM = (40, 52, 2, 40,52,2,80),EPISODES=20000,START_BOX2=(585, 105),START_BOX1=(600, 300), START_BASE=(600, 500),
                      space=space,Q_TABLE_FILE=Q_TABLE_FILE, is_train=True)
    env.set_reward_param(0.2, 0.8)
    pygame.display.set_caption(Q_TABLE_FILE)
    env.execEnv()
    