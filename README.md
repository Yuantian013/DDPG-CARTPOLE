# DDPG_CARTPOLE
Stable control a cartpole with DDPG in continuous actions

# Environment Description
We use OpenAI's cartpole, but make its actions continuous.
And there are many noise in this environment setting, but our policy is still very robust.
In every 0.02s, the Cart's mass changes in a gaussian distribution (1,0.2).
In every 0.02s, the Pole's mass changes in a gaussian distribution (0.1,0.02).
In every 0.02s, the gravity changes in a gaussian distribution (10,2).

# Dependencies
- Tensorflow (1.9.0)
- OpenAi gym (0.10.8)

 ## Reference
[1] [Reinforcement-learning-with-tensorflow](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)  

