# DDPG_MountainCar


The mountain car continuous problem from gym was solved using DDPG, with neural networks
as function aproximators.
The solution is inspired in the DDPG algorithm, but using only low level information as inputs to the net, basically the net uses the position and velocity from the gym environment.
The exploration is done by adding Ornstein-Uhlenbeck Noise to the process. 


## Requirements:

- Numpy
- Tensorflow
- Open AI Gym


## How to run

There is a Constant DEVICE = '/cpu:0', you if you have a gpu you can set it to DEVICE = '/gpu:0' and it will use tensorflow for training. 
To run the algorithm you can do: 

```
python mountain.py
```

If there is a model saved in the folder it will load and start the training/testing. For testing set episilon = 0.   

## Sources:

- [gym Mountain car Continuous](https://github.com/openai/gym/wiki/MountainCarContinuous-v0)
- [sutton's book.](http://people.inf.elte.hu/lorincz/Files/RL_2006/SuttonBook.pdf) 
- [DDPG Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
- [Pemami's blog](http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html) 
- [Implementation of the Ornstein-Uhlenbeck Noise](https://github.com/openai/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py)
- [Blog about RL](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2) 
- [Playing Torch w/ keras](https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html) Good explanation of how everything works.
