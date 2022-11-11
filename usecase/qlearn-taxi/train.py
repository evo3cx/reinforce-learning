from tqdm import tqdm
import numpy as np
from policy import Policy

class Trainer:
  def __init__(self, policy: Policy) -> None:
    self.policy = policy

  def train(self, n_training_episodes, min_epsilon, max_epsilon, learning_rate, gamma, decay_rate, max_steps, Qtable):
    print('n_train_episodes', n_training_episodes)
    for episode in tqdm(range(n_training_episodes)):
      # Reduce epsilon (because we need less and less exploration)
      epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
      # Reset the environment
      state = self.policy.env.reset()
      step = 0
      done = False

      # repeat
      for step in range(max_steps):
        # Choose the action At using epsilon greedy policy
        action = self.policy.epsilon_greedy_policy(Qtable, state, epsilon)

        # Take action At and observe Rt+1 and St+1
        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = self.policy.env.step(action)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        Qtable[state][action] = Qtable[state][action] + learning_rate * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state, action])

        # If done, finish the episode
        if done:
          break
        
        # Our state is the new state
        state = new_state
    return Qtable

  def evaluate_agent(self, max_steps, n_eval_episodes, Q, seed):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param Q: The Q-table
    :param seed: The evaluation seed array (for taxi-v3)
    """
    episode_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
      if seed:
        state = self.policy.env.reset(seed=seed[episode])
      else:
        state = self.policy.env.reset()
      step = 0
      done = False
      total_rewards_ep = 0
      
      for step in range(max_steps):
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(Q[state][:])
        new_state, reward, done, info = self.policy.env.step(action)
        total_rewards_ep += reward
          
        if done:
          break
        state = new_state
      episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward