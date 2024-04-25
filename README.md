# GSP-RL
Library for Multi-Agent Learning

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
