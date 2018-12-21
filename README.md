# DDPG_CARTPOLE
-  Stable and robust control a cartpole in continuous actions with large noise by using DDPG.

# Environment Description
-  We use OpenAI's cartpole, but make its actions continuous.
-  And there are many noise in this environment setting, but our policy is still very robust.
## Internal uncertainty
-  In every 0.02s, the Cart's mass changes in a gaussian distribution (1,0.2).
-  In every 0.02s, the Pole's mass changes in a gaussian distribution (0.1,0.02).
-  In every 0.02s, the gravity changes in a gaussian distribution (10,2).
## Action uncertainty
-  And the action the agent chooses will also be added with a gaussian distribution(action,10).
-  The torch, the acceleration，angular acceleration all add with a gaussian distribution.

## Model
- cartploe_normal.ckpt train with no uncertainty.
- cartploe_plus_5.ckpt train with full uncertainty.

## Env
- cartpole_env.py is without uncertainty environment.
- cartpole_plus.py is the uncertainty environment.

# Dependencies
- Tensorflow (1.9.0)
- OpenAi gym (0.10.8)

 ## Reference
[1] [Reinforcement-learning-with-tensorflow](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)  

