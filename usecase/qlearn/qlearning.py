from qlearn_env import Env
import numpy as np
import time 
import os
import pandas as pd

# create a environemtn
env = Env()

# QTable : contains the Q-Value for every (state, action) pair
qtable = np.random.rand(env.stateCount, env.actionCount).tolist()

# hyperparameters
epochs = 30
gamma = 0.1
epsilon = 0.08 #epsilon_greedy
decay = 0.1

# training loop
for i in range(epochs):
    state, reward, done = env.reset()
    steps = 0

    while not done:
        try: 
            os.system('clear')

            print("epoch #", i+1, "/", epochs)
            env.render()
            time.sleep(0.05)

            # count steps to finish game
            steps += 1

            # act randomly sometimes to allow exploration
            if np.random.uniform() < epsilon:
                action = env.randomAction()
            # if not select max action in Qtable (act greedy)
            else:
                action = np.argmax(qtable[state])
            
            # take action
            next_state, reward, done = env.step(action)

            # update qtable value with Bellman equation
            qtable_next = qtable[next_state]
            qtable[state][action] = reward + gamma * max(qtable_next)

            # update state
            state = next_state
        except Exception as e:
            print("next state", next_state)
            print("qtable", len(qtable))
            raise e
    
    # the more we learn, the less we take random actions
    epsilon -= decay * epsilon

    print("\nDone in", steps, "steps".format(steps))
    time.sleep(0.8)

qtable_df = pd.DataFrame(qtable)
qtable_df.sum(axis=1).to_csv('qtable_step_reward.csv')

