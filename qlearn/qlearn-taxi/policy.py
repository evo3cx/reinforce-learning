import numpy as np
import gym

class Policy:
  def __init__(self) -> None:
    # Create the FrozenLake-v1 environment using 4x4 map and non-slippery version
    self.env = gym.make("Taxi-v3")

  # Let's create our Qtable of size (state_space, action_space) and initialized each values at 0 using np.zeros
  def initialize_q_table(self,state_space, action_space):
    Qtable = np.zeros((state_space, action_space))
    return Qtable


  def epsilon_greedy_policy(self, Qtable, state, epsilon):
    # Randomly generate a number between 0 and 1
    random_num = np.random.rand(0, 1)
    # if random_num > greater than epsilon --> exploitation
    if random_num > epsilon:
      # Take the action with the highest value given a state
      # np.argmax can be useful here
      action = np.argmax(Qtable[state])
    # else --> exploration
    else:
      # Take a random action
      action = self.env.action_space.sample()
  
    return action
  
  def greedy_policy(self, Qtable, state):
    # Exploitation: take the action with the highest state, action value
    action = np.argmax(Qtable[state])
    
    return action
  
  def get_env_space(self, verbose = True):
    state_space = self.env.observation_space.n
    action_space = self.env.action_space.n

    if verbose == True:
      print("\n _____ACTION SPACE_____ \n")
      print("Action Space Shape", self.env.action_space.n)
      print("Action Space Sample", self.env.action_space.sample()) # Take a random action

      print("\n _____OBSERVATION SPACE_____ \n")

      print("There are ", state_space, " possible states")

      print("There are ", action_space, " possible actions")
    
    return state_space, action_space