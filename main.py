from PendulumEnv import PendulumEnv
from PendulumEnv import space
import pygame
"""
Instruction for use:
    - To execute for training set is_train to True in the initialization of the environment, False for simulate;
    - In any case is necessary specify the file name on which save the table (variable Q_TABLE_FILE);
    - To set reward parameters (alpha, beta) use the set_reward_param function 
"""
if __name__ == "__main__":
    Q_TABLE_FILE ="Tests/table1.json"
    is_train = False
    env = PendulumEnv(LEARNING_RATE = 0.1, DISCOUNT=0.95, MAX_EPSILON=1.0, MIN_EPSILON=0.05, 
                      Q_TABLE_DIM = (40, 20, 2, 20),EPISODES=25000,START_BOX=(600, 500), START_BASE=(600, 300),
                      space=space,Q_TABLE_FILE=Q_TABLE_FILE, is_train=is_train)
    env.set_reward_param(0.5, 0.5)
    pygame.display.set_caption(Q_TABLE_FILE)
    env.execEnv()
    