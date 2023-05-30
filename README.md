# Self Blancing AI
## Project Structure
  The code is structured as follow:
  - Grpahics.py contains all the classes for the graphic visualizzation;
  - PendulumEnv.py contains the main function and the class for the Environment training and simiulation;
  - Tests folder contains some results of the experiments such as plot of the functions, recording of some interesting results and some tables;
  - cmd.txt is the file thanks to which the user can give command to the program;
  - log.txt is the log file of the trainig session that were executed;
## Instructions for use
  To execute a training session set is_train to True in the main.py file, False for simulate a table.
  In any case is necessary specify the file name which will be used to save the final table, in case of training execution, and to simulate an existing table, in case of simulation.
  It's possible to set the Environment parameters by changing the values in the PendulumEnv initialization (main.py).
  set_reward_param function could be use to change the reward weight (alpha and beta).