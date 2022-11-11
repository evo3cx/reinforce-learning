This code help me to understand the code behind Math of Q-Learning.

# Introducing Q-Learning

### What is Q-Learning?

Q-Learning is an off-policy value-based method that uses a TD approach to train its action-value function:

- Off-policy: we'll talk about that at the end of this chapter.
- Value-based method: finds the optimal policy indirectly by training a value or action-value function that will tell us the value of each state or each state-action pair.
- Uses a TD approach: updates its action-value function at each step instead of at the end of the episode.

## Repository:

1. FrozenLake v1: https://huggingface.co/reza-aditya/q-FrozenLake-v1-4x4-noSlippery

References:

1. https://towardsdatascience.com/math-of-q-learning-python-code-5dcbdc49b6f6
2. https://huggingface.co/blog/deep-rl-q-part1