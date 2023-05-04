import cv2
import numpy as np
import time
import pymunk               # Import pymunk..
import pygame
import json
import sys


pygame.init()
display = pygame.display.set_mode((1200,600))
clock = pygame.time.Clock()
space = pymunk.Space()
FPS = 50

xcamera = 0
ycamera = 300

class Box():
    def __init__ (self, x ,y,width,height,density = 1,  static=False, color =(0,0,0)):
        if static:
            self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        else:
            self.body = pymunk.Body()
        self.body.position = x, y
        self.width = width
        self.height = height
        self.shape = pymunk.Poly.create_box(self.body, (width, height))
        self.shape.density = density
        space.add(self.body, self.shape)
        self.color = color
    def moveX(self,offset):
        self.body.velocity = (offset, 0)
    def draw(self):
        x, y = self.body.position
        pygame.draw.rect(display, self.color,(int(x-self.width/2) - xcamera, int(y- self.height/2) , self.width, self.height))

class String():
    def __init__(self, body1, attachement):
        self.body1 = body1
        self.body2 = attachement
        joint = pymunk.PinJoint(self.body1, self.body2)
        space.add(joint)
    def draw(self):
        x1, y1 = self.body1.position
        x2, y2 = self.body2.position
        pygame.draw.line(display,(0,0,0), (int(x1)-xcamera, int(y1)), (int(x2)-xcamera, int(y2)), 2)

class PendulumEnv:
    def __init__(self, LEARNING_RATE, DISCOUNT, MAX_EPSILON, MIN_EPSILON, DEACY_RATE, Q_TABLE_DIM,EPISODES, ACTION_NUM, base, box, string,space):
        self.LEARNING_RATE = LEARNING_RATE
        self.DISCOUNT = DISCOUNT
        self.MAX_EPSILON = MAX_EPSILON
        self.MIN_EPSILON = MIN_EPSILON
        self.DECAY_RATE = DEACY_RATE
        self.EPISODES = EPISODES
        self.ACTION_NUM = ACTION_NUM
        self.MAX_FRAME = 1000

        self.q_table = np.zeros(Q_TABLE_DIM)
        self.base = base
        self.box = box
        self.string = string
        self.prev_pos = [0,0]
        self.timer = 0
        self.space = space
        self.space.gravity = (0, 1000)
        self.action = 0
        self.frame = 0

    def get_epsilon(self,alpha):
        return self.MAX_EPSILON - (self.MAX_EPSILON /self.EPISODES) * alpha
        #return self.MIN_EPSILON + (self.MAX_EPSILON - self.MIN_EPSILON)*np.exp(-self.DECAY_RATE*(alpha))
    
    def get_reward(self):
        alpha = 0.8
        dir = self.UP_or_DOWN()
        v = self.action
        if dir == 0:
            dir = -1
        angle = self.get_angle()
        distance_from_x = 0
        if angle < 90:
            distance_from_x =  angle//18 + 5
        elif angle < 270: 
            distance_from_x = 15 - angle//18
        elif angle > 270:
            distance_from_x = angle//18 - 15
        return alpha*dir + (1-alpha) * distance_from_x/10
        return alpha * dir + (1-alpha)*((40-np.abs(v))/40)
        """
        angle = int(self.get_angle()//18)
        if self.UP_or_DOWN() == 1 and  not((self.pre_angle >= 4 and self.pre_angle <= 5) and (angle <4 or angle >5)):
            return 1
        return -1"""

    def UP_or_DOWN(self):
        if self.box.body.position[1] > self.prev_pos[1]:
            return 0 #DOWN
        return 1 #UP
        
    def get_angle(self):
        xbox, ybox = self.box.body.position
        xbase, ybase = self.base.body.position
        if xbase == xbox:
            if ybox-ybase > 0:
                return 227
            else:
                return 90
        mr = np.abs(-(ybox - ybase)/(xbox - xbase))
        def from_0_to_360(x, y, xo, yo, angle):
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
        angle = self.get_angle()
        return int((angle - angle%18)/18), self.get_discrete_velocity(self.get_continuos_velocity(self.box.body.velocity)), self.UP_or_DOWN() 

    def get_continuos_velocity(self, velocity):
        v1, v2 = velocity
        return np.sqrt(v1*v1 + v2*v2)
        
    def get_discrete_velocity(self, velocity):
        MAX_VEL = 620
        discrete_v = min(velocity, MAX_VEL-1)
        if discrete_v <= 10:
            return int(discrete_v)
        elif discrete_v <= 40:
            return int(11 +  (discrete_v-10)//5)
        elif discrete_v <= 100:
            return int(17 + (discrete_v-40)//10)
        elif discrete_v <= 220:
            return int(23 + (discrete_v-100)//20)
        elif discrete_v <= 420:
            # TODO Unire  ultimi due casi
            return int(28 + (discrete_v-220)//40)
        elif discrete_v <= MAX_VEL:
            return int(33 + (discrete_v-420)//40)
        #return int(discrete_v)
        return int((discrete_v) // np.ceil(MAX_VEL/80))

    def episode_status(self):
        state = self.get_new_state()
        
        if state[0] >= 4 and state[0] <= 5:
            self.box.color = (0,255,0)
            if (time.time()-self.timer) > 10:
                return (True, False)
            
        elif state[0] >= 14 and state[0] <= 15: 
            self.box.color = (255, 0,0)
            self.timer = time.time()
            return (False, self.frame >= self.MAX_FRAME)
        else:
            self.timer = time.time()
            self.box.color = (191, 64, 191)
        return (False, False) 

    def step(self,action):
        self.prev_pos = self.box.body.position
        self.action = action
        self.base.moveX(action) 
        self.space.step(1/FPS)
        return self.get_reward(), self.get_new_state(), self.episode_status()[0],self.episode_status()[1]
 
    def sample_cond(self, i):
        return i == (self.EPISODES -1)
    def train(self):
        global xcamera
        global ycamera
        cmd_t = 0
        for episode in range(self.EPISODES):
            epsilon = self.get_epsilon(episode)
            done = False
            render = False
            truncated = False
            state = (0,0,0)
            space.remove(self.base.shape, self.base.body)
            space.remove(self.box.shape, self.box.body)
            xcamera = 0
            ycamera = 300
            self.base= Box(600,300, 100, 10, static=True)
            self.box = Box(599,500, 50, 50, color=(191, 64, 191))
            self.string = String(self.base.body, self.box.body)
            self.frame = 0
            print(episode)
            if self.sample_cond(episode):
                input("Last episode")
            line = ''
            with open('cmd.txt', 'r')as cmd:
                line = cmd.readline()
                if line: 
                    if line == 'save' and not(cmd_t == 1):
                        print('Saving...', cmd_t)
                        with open('q_table.json','w') as f:
                            tosave =[]
                            x,y,z,d = self.q_table.shape
                            for i in range(x):
                                for j in range(y):
                                    for q in range(z):
                                        for w in range(d):
                                            tosave.append(self.q_table[i][j][q][w])
                            json.dump(tosave, f)
                        cmd_t = 1
                    if line == 'exit':
                        cmd_t= 2
                        break
                    if line == 'show':
                        cmd_t = 3
                        render = True
                else:
                    cmd_t = 0
                    
            self.timer = time.time()
            while not done and not truncated:
                if np.random.random() > epsilon:
                    action = np.argmax(self.q_table[state[0],state[1],state[2]])
                else:
                    action = np.random.randint(0, self.ACTION_NUM)
                speed = action%(self.ACTION_NUM//2) * 17
                if action > (self.ACTION_NUM//2):
                    speed = -speed

                reward,new_state, done, truncated = self.step(speed)
                if done:
                    print(episode, done)
                if self.sample_cond(episode) or render:
                    self.render()
                max_future_q = np.max(self.q_table[new_state[0],new_state[1],new_state[2]])
                current_q = self.q_table[state[0]][state[1]][state[2]][action]
                new_q = (1-self.LEARNING_RATE) * current_q + self.LEARNING_RATE * (reward + self.DISCOUNT * max_future_q)
                
                self.q_table[state[0],state[1],state[2]][action] = new_q
                state = new_state
                self.frame += 1
            print("Frame: ", self.frame)
        with open('q_table.json','w') as f:
            tosave =[]
            x,y,z,d = self.q_table.shape
            for i in range(x):
                for j in range(y):
                    for q in range(z):
                        for w in range(d):
                            tosave.append(self.q_table[i][j][q][w])
            json.dump(tosave, f)
        
    def load_q_table(self, file):
        q_list =[]
        with open(file,'r') as f:
            q_list = json.load(f)
        ind = 0
        q_table = np.zeros(self.q_table.shape)
        x,y,z,d = self.q_table.shape
        for i in range(x):
            for j in range(y):
                for q in range(z):
                    for w in range(d):
                        q_table[i][j][q][w] = q_list[ind]
                        ind += 1
        return q_table
    def simulate(self):
        input("Start")
        self.q_table = self.load_q_table('q_table.json')
        done = False
        truncated = False
        state = (0,0,0)
        self.timer = time.time()
        space.remove(self.base.shape, self.base.body)
        space.remove(self.box.shape, self.box.body)
        self.base= Box(600,300, 100, 10, static=True)
        self.box = Box(599,100, 50, 50, color=(191, 64, 191))
        self.string = String(self.base.body, self.box.body)
        while  not truncated:
            action = np.argmax(self.q_table[state[0],state[1],state[2]])
            speed = action%(self.ACTION_NUM//2)
            if action > (self.ACTION_NUM//2):
                speed = -speed
            reward,new_state, done, truncated = self.step(speed)
            self.render()
            state = new_state
    def render(self):
        global xcamera
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
        txt = pygame.font.SysFont("Arial", 30).render(str(time.time()- self.timer)[:3], True, (0,0,0))
        
        display.fill((255,255,255))
        display.blit(txt, (30, 30))
        self.base.draw()
        self.box.draw()
        self.string.draw()
        if self.base.body.position[0] -(xcamera+600)> 600:
            xcamera += 1200
        if (xcamera+600)- self.base.body.position[0]> 600:
            xcamera -= 1200
        pygame.display.update()
        clock.tick(FPS)

if __name__ == "__main__":
    base = Box(600,300, 100, 10, static=True)
    box = Box(550,50, 50, 50, color=(191, 64, 191))
    string = String(base.body, box.body)
    env = PendulumEnv(LEARNING_RATE = 0.7, DISCOUNT=0.95, MAX_EPSILON=1.0, MIN_EPSILON=0.05, DEACY_RATE=0.0005, Q_TABLE_DIM = (20, 80, 2, 80),EPISODES=20000, ACTION_NUM=80, base=base, box= box, string=string,space=space)
    env.train()
    #floor = Box (300, 350, 800,10, static=True)
    