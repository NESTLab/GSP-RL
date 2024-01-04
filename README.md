# GSP-RL
Library for Multi-Agent Learning

## Setup Guide

Before getting started, you will need [pyenv](https://github.com/pyenv/pyenv) and [poetry](https://python-poetry.org/docs/) installed on your machine

1. Use pyenv to install and use python `3.10.2`.

Instalition Notes:
- You may run into an issue with the library `_tkinter`, I found installing this specific version solved the problem:
```sudo apt-get install tk-dev```

2. Specify the python version that poetry should use and create a `virtualenv`.
 ```
 poetry env use 3.10.2
 ```

3. Install the package and its dependancies

```
poetry intsall
```
