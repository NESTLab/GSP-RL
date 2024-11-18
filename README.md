# GSP-RL
GSP-RL is a library built on PyTorch for several deep learning predictive methods to eliminate non-stationarity in MARL. This method uses A-CTDE which assumes and accounts for agent's actions impacts on every other agent's state after action execution due to either rigid or soft lattice formations. 

![Training_Schemes](https://github.com/user-attachments/assets/801f55b7-fa47-491a-9d1a-501e0b4fe9e7)


This Library supports 4 common RL Algorithms:
1. DQN      http://www.nature.com/articles/nature14236
2. DDQN     https://arxiv.org/abs/1509.06461
3. DDPG     http://arxiv.org/abs/1509.02971
4. TD3      https://arxiv.org/abs/1802.09477

A study of these 4 algorithms as it pertains to multi-agent reinforcement learning in swarm robotics, specifically collective transport with imperfect robots, can be found here: https://arxiv.org/pdf/2203.15129

To address the issues identified in the study above, we introdice Global State Prediction (GSP). GSP is a decentralized predictive network that observes partial observations over the other agents in the swarm and provides a prediction on the global state in the next time step. This prediction is a direct result of the actions to be executed, thus giving each agent a prediction on what the rest of the collective will do. The prediction is then fed into the action network as part of the observation at the current time step. We present GSP here: https://scholar.google.com/citations?view_op=view_citation&hl=en&user=r_6eZtMAAAAJ&citation_for_view=r_6eZtMAAAAJ:UeHWp8X0CEIC

![GSP](https://github.com/user-attachments/assets/12e12649-2a90-4344-8916-d326736ef0e4)

We next reduce the communication size of GSP to limit it to its immediate neighborhood. In our Collective Transport example this results in the immediate neighbors clockwise and counterclockwise of the current agent. This allows us to greatly reduce the scale of communication as the swarm grows and allows us to generalize a method trained on a specific number of robots to any number of robots. These results are to be submitted and a link to the paper will be uploaded shortly. 

[bandwidth (2).pdf](https://github.com/user-attachments/files/17801316/bandwidth.2.pdf)


We next study the role memory plays in a distributed system. We introduce two new variations of GSP. Firstly we append a Reccurent Neural Network to the front end of GSP in the form of a LSTM layer. This provides initial short term memory retention of relevant near term history while providing overarching memory of longer term events. This is esspecially important when coming in contact with obstacles in the environment. We term this version of GSP as RGSP.

![RGSP](https://github.com/user-attachments/assets/d1262239-66ea-46d2-902e-9992f4755536)


Next, we study a novel implementation of Attention encoding by replacing the GSP architecture with an attention encoder with modified front and back end to allow for continuous floating point numbers to be passed as input and for a normalized prediction to be output. We term this version, A-GSP

![A-GSP - Page 1](https://github.com/user-attachments/assets/fe5de09b-fb6d-4424-8350-6798bed22d7b)

![AGSP](https://github.com/user-attachments/assets/6cc2b223-b487-4908-8bdb-6b13a49ad9b2)

Both R-GSP and A-GSP are to be submitted shortly and a link will be uploaded. 


## Setup Guide

Before getting started, you will need [pyenv](https://github.com/pyenv/pyenv) and [poetry](https://python-poetry.org/docs/) installed on your machine

After you install pyenv, you will need to add the following to `.bashrc` and restart your terminal
```
# pyenv
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

1. Use pyenv to install and use python `3.10.2`.
```
pyenv install 3.10.2
```

2. Create a new virtual dev environment
```
pyenv virtualenv 3.10.2 gsprl
```

3. Avtivate the new virtual environment
```
pyenv activate gsprl
```

4. Update pip
```
python3.10 -m pip install --upgrade pip
```

5. Install Poetry
```
pip install poetry
```

Instalition Notes:
- You may run into an issue with the library `_tkinter`, I found installing this specific version solved the problem:
```sudo apt-get install tk-dev```

6. Specify the python version that poetry should use and create a `virtualenv`.
 ```
 poetry env use 3.10.2
 ```

7. Install the package and its dependancies

```
poetry install
```

## Testing
1. Unit Tests: you can run unit tests via the command
```
poetry run pytest
```

2. RL Testing: you can test the RL algorithms on several different gym environments via the examples directory
```
$ cd examples/baselines
$ python cart_pole.py
$ python lunar_lander.py
$ python pendulum.py
```

CartPole and LunarLander are Discrete action space environments and thus can be learned via DQN or DDQN. Pendulum is a Coninuous action space and thus can be learned via DDPG or TD3
