# Virtual display
from pyvirtualdisplay import Display
import datetime
import json

import pickle5 as pickle

# internal lib
from video import record_video
from policy import Policy
from train import Trainer

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

# Create policy with environemtn gym Taxi
policy = Policy()

# print env info
state_space, action_space = policy.get_env_space()

# Create our Q table with state_size rows and action_size columns (500x6)
Qtable_taxi = policy.initialize_q_table(state_space, action_space)

# HYPERPARAMETERS
# Training parameters
n_training_episodes = 25000   # Total training episodes
learning_rate = 0.7           # Learning rate

# Evaluation parameters
n_eval_episodes = 100        # Total number of test episodes

# DO NOT MODIFY EVAL_SEED
eval_seed = [16,54,165,177,191,191,120,80,149,178,48,38,6,125,174,73,50,172,100,148,146,6,25,40,68,148,49,167,9,97,164,176,61,7,54,55,
 161,131,184,51,170,12,120,113,95,126,51,98,36,135,54,82,45,95,89,59,95,124,9,113,58,85,51,134,121,169,105,21,30,11,50,65,12,43,82,145,152,97,106,55,31,85,38,
 112,102,168,123,97,21,83,158,26,80,63,5,81,32,11,28,148] # Evaluation seed, this ensures that all classmates agents are trained on the same taxi starting position
                                                          # Each seed has a specific starting state

# Environment parameters
env_id = "Taxi-v3"           # Name of the environment
max_steps = 99               # Max steps per episode
gamma = 0.95                 # Discounting rate

# Exploration parameters
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.05           # Minimum exploration probability 
decay_rate = 0.005            # Exponential decay rate for exploration prob

# START TRAIN
trainer = Trainer(policy)
Qtable_taxi =  trainer.train(n_training_episodes, min_epsilon, max_epsilon, learning_rate, gamma, decay_rate, max_steps, Qtable_taxi)

# BUILD MODEL
model = {
  "env_id": env_id,
  "max_steps": max_steps,
  "n_training_episodes": n_training_episodes,
  "n_eval_episodes": n_eval_episodes,
  "eval_seed": eval_seed,

  "learning_rate": learning_rate,
  "gamma": gamma,

  "max_epsilon": max_epsilon,
  "min_epsilon": min_epsilon,
  "decay_rate": decay_rate,

  "qtable": Qtable_taxi
}

# Step 1: Save the model
env = policy.env
if env.spec.kwargs.get("map_name"):
  model["map_name"] = env.spec.kwargs.get("map_name")
  if env.spec.kwargs.get("is_slippery", "") == False:
      model["slippery"] = False

  print("model \n", model)

# Pickle the model
with open('taxi-v3-q-learning.pkl', 'wb') as f:
  pickle.dump(model, f)

# Step 2: Evaluate the model and build JSON
eval_env = env
mean_reward, std_reward = trainer.evaluate_agent(
                                    model["max_steps"], 
                                    model["n_eval_episodes"], 
                                    model["qtable"], 
                                    model["eval_seed"])

# First get datetime
eval_datetime = datetime.datetime.now()
eval_form_datetime = eval_datetime.isoformat()

evaluate_data = {
  "env_id": model["env_id"], 
  "mean_reward": mean_reward,
  "n_eval_episodes": model["n_eval_episodes"],
  "eval_datetime": eval_form_datetime,
}

# Write a JSON file
with open("results.json", "w") as outfile:
  json.dump(evaluate_data, outfile)


  # Step 4: Record a video
print('start record a video')
video_path =  "replay.mp4"
video_fps = 1
record_video(env, model["qtable"], video_path, video_fps)
print('finish record a video')