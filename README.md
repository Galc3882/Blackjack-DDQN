# Blackjack-DDQN

## Introduction

Blackjack-DDQN is a project that uses deep reinforcement learning to train a Double Q-learning model on a Blackjack game environment. The project is implemented using TensorFlow in Python.

The goal of this project is to demonstrate how to use deep reinforcement learning algorithms to train agents to play simple games like Blackjack. The project is also designed to be an educational resource for people who are interested in learning about deep reinforcement learning.

## Table of Contents

- [Explanation of Project](#explanation-of-project)
- [Special Considerations](#special-considerations)
- [Examples](#examples)
- [Requirements and File Architecture](#requirements-and-file-architecture)
- [Instructions](#instructions)
- [Results](#results)

## Explanation of Project

Blackjack-DDQN is a reinforcement learning model that learns to play the game of Blackjack. The model is implemented using a Double Q-learning algorithm, which helps to reduce overestimation of action values. The agent learns to make decisions based on the current state of the game and the available actions, with the goal of maximizing its long-term reward.

The model is trained on a simulated Blackjack game environment. The environment provides the agent with the necessary information to make decisions, such as the dealer's up-card and the player's hand.

The TensorFlow framework is used to implement the deep neural network used in the model. The neural network takes the current state of the game as input and outputs the estimated values of each action. The agent uses these values to make decisions about which action to take next.

## Special Considerations

One of the challenges of training a reinforcement learning agent on a game environment is the balance between exploration and exploitation. The agent must explore the game environment to learn how to play effectively, but it must also exploit its current knowledge to maximize its reward.

To address this challenge, the model uses an epsilon-greedy exploration strategy. During training, the agent randomly selects actions with a certain probability, allowing it to explore the game environment. As training progresses, the probability of exploration decreases, and the agent relies more heavily on its learned knowledge to make decisions.

## Examples

![image](https://user-images.githubusercontent.com/86870298/124348614-86cc3d80-dbf3-11eb-895b-38f3421b2d15.png)
![image](https://user-images.githubusercontent.com/86870298/124348579-4f5d9100-dbf3-11eb-9123-a7dd75913147.png)

## Requirements and File Architecture

### Imports

The project uses the following Python packages:

- TensorFlow
- NumPy

### File Architecture

## File Architecture
```
├── data/
│   ├── tf/
│   │   ├── checkpoint
│   │   └── checkpoint.old
├── README.md
├── game without DQN.py
└── game.py
```

- `data/tf/`: Directory where TensorFlow checkpoints are saved during training
- `data/tf/checkpoint`: TensorFlow checkpoint file
- `data/tf/checkpoint.old`: TensorFlow checkpoint backup file
- `README.md`: Readme file with project information
- `game without DQN.py`: Python file containing the implementation of a Blackjack game without DQN
- `game.py`: Python file containing the implementation of a Blackjack game with DQN. This file uses the trained DQN model to play the game.

## Instructions
To run the game without DQN, execute the following command:
```
python game_without_DQN.py
```
This will start the game without the DQN algorithm.

To run the game with DDQN, navigate to the project directory and run the following command:
```
python game.py
```
This will start the game with the DQN algorithm. The game will load the trained DQN model from the TensorFlow checkpoint files in the `data/tf` directory.

## Results
The Blackjack-DDQN project successfully implemented a Double Deep Q-Learning algorithm to train a neural network on the Blackjack game environment using TensorFlow in Python.

The trained model achieved a loss ratio of 87.1%, which is close to the theoretical value of 99.4%. This result demonstrates the effectiveness of using a Deep Reinforcement Learning approach for training a model on this task.

This project can serve as a useful reference for anyone interested in using Deep Reinforcement Learning for training a model on the Blackjack game environment or similar tasks. The project also highlights the importance of Double Q-Learning in overcoming the overestimation bias often encountered in Q-Learning algorithms.

Next steps for this project could include further optimizations and improvements to the model, as well as exploring its performance on other game environments.
