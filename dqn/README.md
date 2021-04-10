# Deep Q-Learning
This directory implements the individual components of [Rainbow](https://arxiv.org/abs/1710.02298) into
separate scripts, so each improvement on the basic DQN can be tested individually.

Once all the individual components have been implemented, they will be merged into the full, Rainbow
architecture.

This directory is heavily based off Chapter 8 in [Deep Reinforcement Learning Hands-On](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition) by Maxim Lapan with significant divergence where it is appropriate (because what is the point in blindly copying?). In particular, I placed significant emphasis on engineering the code to easily support changes to experimental variables via config files.

## TODO
- [X] Basic DQN
- [X] Double DQN
- [X] Dueling DQN
- [X] N-step rewards
- [X] Noisy DQN
- [X] Prioritized Replay Buffers
- [ ] Categorical DQN
- [ ] Rainbow (Combining Everything)

### Citations

Rainbow: Combining Improvements in Deep Reinforcement Learning
```
@misc{hessel2017rainbow,
      title={Rainbow: Combining Improvements in Deep Reinforcement Learning}, 
      author={Matteo Hessel and Joseph Modayil and Hado van Hasselt and Tom Schaul and Georg Ostrovski and Will Dabney and Dan Horgan and Bilal Piot and Mohammad Azar and David Silver},
      year={2017},
      eprint={1710.02298},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
