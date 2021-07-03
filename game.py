import pygame
import random
import time
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile
import logging
import tensorboard
from datetime import datetime

import tensorflow as tf
tf.get_logger().setLevel(logging.WARNING)

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import q_network
from tf_agents.policies import policy_saver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.train.utils import strategy_utils

#%%
# Functions for the game
def showtext(text, x, y, color = []):
    screen.blit(font.render(str(text), True, color), (x, y))
    #pygame.display.update()
def clearscreen():
    screen.fill((0, 0, 0))
    #pygame.display.update()
def drawcardpack():
    cardpack = []
    numofpacks = 8
    while numofpacks > 0:
        i = 1
        while i < 14:
            cardpack.extend([i, i, i, i])
            i += 1
        numofpacks -= 1
    return cardpack
def initgame(playercards, dealercards, cardpack):
    playercards.clear()
    dealercards.clear()
    card, cardpack = getnextcard(cardpack)
    playercards.append(card)
    card, cardpack = getnextcard(cardpack)
    dealercards.append(card)
    card, cardpack = getnextcard(cardpack)
    playercards.append(card)
    card, cardpack = getnextcard(cardpack)
    dealercards.append(card)
    card, cardpack = getnextcard(cardpack)
    return playercards, dealercards, cardpack
def isnatural(cardlist):
    return max(totalcards(cardlist)) == 21
def playerlose(score, color):
    showtext('player lost', width*0.8, hight*0.05, color)
    start, running = asknewgame(color)
    return start, running, score - 1
def tie(color):
    showtext('tie', width*0.8, hight*0.05, color)
    start, running = asknewgame(color)
    return start, running
def asknewgame(color):
    lastkeypressed = None
    running = True
    #time.sleep(0.5)
    showtext('restart? [Y]', width*0.45, hight*0.9, color)
    #time.sleep(0.5)
    while (lastkeypressed != pygame.K_y and running == True):
        lastkeypressed = pygame.K_y
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                lastkeypressed = event.key
            if event.type == pygame.QUIT:
                running = False
                break
    clearscreen()
    start = True
    return start, running
def getnextcard(cardpack):
    if len(cardpack) == 0:
        cardpack = drawcardpack()
    return cardpack.pop(random.randrange(0, len(cardpack), 1)), cardpack
def totalcards(cardlist):
    k = 0
    for i in cardlist:
        if i > 10:
            k += 10
        else:
            k += i
    if cardlist.count(1) > 0 and k + 10 < 22:
        return [k, k + 10]
    return [k, 0]
def showcard(card):
    if card > 10:
        if card == 11:
            return 'J'
        elif card == 12:
            return 'Q'
        elif card == 13:
            return 'K'
    elif card == 1:
        return 'A'
    return card 
def drawstarttext(playercards, score, color):
    drawplayercards(playercards, color)
    drawscore(score, color)
def drawfirstdealercard(dealercards, color):
    showtext('dealers cards: ' + str(showcard(dealercards[0])), width*0.3, hight*0.35, color)
def drawplayercards(playercards, color):
    showtext('players cards: ' + str(showcard(playercards[0])) + ', ' + str(showcard(playercards[1])), width*0.3, hight*0.65, color)
    showtext('sum: ' + str(totalcards(playercards)[0]), width*0.3, hight*0.7, color)
    if totalcards(playercards)[1] != 0:
        showtext(', ' + str(totalcards(playercards)[1]), width*0.4, hight*0.7, color)
def drawnextplayercard(playercards, color):
    showtext(", " + str(showcard(playercards[len(playercards) - 1])), width * (0.54 + 0.035 * (len(playercards) - 3)), hight*0.65, color)
    showtext('sum: ' + str(totalcards(playercards)[0]), width*0.3, hight*0.7, color)
    if totalcards(playercards)[1] != 0:
        showtext(', ' + str(totalcards(playercards)[1]), width*0.4, hight*0.7, color)
def drawnextdealercard(dealercards, color):
    showtext(", " + str(showcard(dealercards[len(dealercards) - 1])), width * (0.54 + 0.035 * (len(dealercards) - 3)), hight*0.35, color)
    showtext('sum: ' + str(totalcards(dealercards)[0]), width*0.3, hight*0.4, color)
    if totalcards(dealercards)[1] != 0:
        showtext(', ' + str(totalcards(dealercards)[1]), width*0.4, hight*0.4, color)
def drawscore(score, color):
    showtext('score: ' + str(score), width*0.05, hight*0.05, color)
def asknextmove(running, cardpack, playercards, isstand, cansplit, issplit, candouble, isdouble, color, action):
    if isdouble == 2:
        hit(cardpack, playercards, color)
        return running, cardpack, playercards, isstand, False, issplit, False, isdouble - 1
    elif isdouble == 1:
        return running, cardpack, playercards, True, False, issplit, False, isdouble
    elif issplit == 1:
        candouble = True
    
    lastkeypressed = -1
    #time.sleep(0.5)
    splitkey = pygame.K_d
    doublekey = pygame.K_d

    if candouble == True and cansplit == True:
        showtext('stand: [A] split: [S] double down: [W] hit: [D]', width*0.2, hight*0.9, color)
        splitkey = pygame.K_s
        doublekey = pygame.K_w
    elif candouble == True:
        showtext('stand: [A] double down: [W] hit: [D]', width*0.3, hight*0.9, color)
        doublekey = pygame.K_w
    else:
        showtext('stand: [A] hit: [D]', width*0.4, hight*0.9, color)

    while (lastkeypressed != pygame.K_a and lastkeypressed != pygame.K_d and lastkeypressed != splitkey and lastkeypressed!= doublekey and running == True):
        if action == 0:
            lastkeypressed = pygame.K_a
        elif action == 1:
            if cansplit == True:
                lastkeypressed = pygame.K_s
            else:
                lastkeypressed = pygame.K_a
        elif action == 2:
            lastkeypressed = pygame.K_d
        elif action == 3:
            if candouble == True:
                lastkeypressed = pygame.K_w
            else:
                lastkeypressed = pygame.K_d

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                lastkeypressed = event.key
            if event.type == pygame.QUIT:
                running = False

    if candouble == True and cansplit == True:
        showtext('stand: [A] split: [S] double down: [W] hit: [D]', width*0.2, hight*0.9, (0, 0, 0))
        if lastkeypressed == pygame.K_s:
            issplit = 2
        elif lastkeypressed == pygame.K_w:
            isdouble = 2
    elif candouble == True:
        showtext('stand: [A] double down: [W] hit: [D]', width*0.3, hight*0.9, (0, 0, 0))
        if lastkeypressed == pygame.K_w:
            isdouble = 2
    else:
        showtext('stand: [A] hit: [D]', width*0.4, hight*0.9, (0, 0, 0))

    if lastkeypressed == pygame.K_a:
        isstand = True
    elif lastkeypressed == pygame.K_d:
        cardpack, playercards = hit(cardpack, playercards, color)
    return running, cardpack, playercards, isstand, False, issplit, False, isdouble
def hit(cardpack, playercards, color):
    showtext('sum: ' + str(totalcards(playercards)[0]), width*0.3, hight*0.7, (0, 0, 0))
    if totalcards(playercards)[1] != 0:
        showtext(', ' + str(totalcards(playercards)[1]), width*0.4, hight*0.7, (0, 0, 0))
    card, cardpack = getnextcard(cardpack)
    playercards.append(card)
    drawnextplayercard(playercards, color)
    return cardpack, playercards
def dealermove(start, running, cardpack, dealercards, score, color):
    drawnextdealercard(dealercards, color)
    while totalcards(dealercards)[1] < 17 and totalcards(dealercards)[1] > 0:
        showtext('sum: ' + str(totalcards(dealercards)[0]), width*0.3, hight*0.4, (0, 0, 0))
        if totalcards(dealercards)[1] != 0:
            showtext(', ' + str(totalcards(dealercards)[1]), width*0.4, hight*0.4, (0, 0, 0))
        card, cardpack = getnextcard(cardpack)
        dealercards.append(card)
        #time.sleep(0.5)
        drawnextdealercard(dealercards, color)
    if not(totalcards(dealercards)[1] > 0):
        while totalcards(dealercards)[0] < 17 and totalcards(dealercards)[1] < 17:
            showtext('sum: ' + str(totalcards(dealercards)[0]), width*0.3, hight*0.4, (0, 0, 0))
            if totalcards(dealercards)[1] != 0:
                showtext(', ' + str(totalcards(dealercards)[1]), width*0.4, hight*0.4, (0, 0, 0))
            card, cardpack = getnextcard(cardpack)
            dealercards.append(card)
            #time.sleep(0.5)
            drawnextdealercard(dealercards, color)
        if totalcards(dealercards)[0] > 21:
            start, running, score = dealerlose(score, color)
    return start, running, cardpack, dealercards, score
def dealerlose(score, color):
    showtext('player win', width*0.8, hight*0.05, color)
    start, running = asknewgame(color)
    return start, running, score + 1
def checkbust(start, running, playercards, score, isdouble, color):
    if totalcards(playercards)[0] > 21:
        start, running, score = playerlose(score, color)
        if isdouble == 1:
            isdouble = 0
            score -= 1
    return start, running, score, isdouble
def showpacksize(lenpack, lastpacksize, color):
    showtext('deck size: ' + str(lastpacksize), width*0.05, hight*0.95, (0, 0, 0))
    showtext('deck size: ' + str(lenpack), width*0.05, hight*0.95, color)
    return lenpack
def split(cardpack, playercards, dealercards, splitcard, splitdealercard, issplit, color):
    drawplayercards(playercards, (0, 0, 0))
    if issplit == 2:
        splitcard = playercards[1]
        splitdealercard.append(dealercards[0])
        splitdealercard.append(dealercards[1])
        playercards.pop(1)
        card, cardpack = getnextcard(cardpack)
        playercards.append(card)
    else:
        cardpack.append(playercards.pop(1))
        cardpack.append(playercards.pop(0))
        cardpack.append(dealercards.pop(1))
        cardpack.append(dealercards.pop(0))
        dealercards.append(splitdealercard[0])
        dealercards.append(splitdealercard[1])
        playercards.append(splitcard)
        card, cardpack = getnextcard(cardpack)
        playercards.append(card)
        splitdealercard = []
        splitcard = 0
    drawplayercards(playercards, color)
    return cardpack, playercards, dealercards, splitcard, splitdealercard, issplit - 1
def checknatural(dealercards, playercards, score, start, running, color, action):
    isinsured = False
    nextafterinsu = False
    if dealercards[0] == 1 and not(isnatural(playercards)):
        isinsured, running = insurance(running, action, color)
        nextafterinsu = True
        if isinsured:
            if not(isnatural(dealercards)):
                score -= 0.5
                showtext('insurance lost', width*0.1, hight*0.5, color)
            else:
                score -= 1
                drawnextdealercard(dealercards, color)
                showtext('insurance won', width*0.1, hight*0.5, color)
                start, running, score = dealerlose(score, color)
    if  running == True:
        if isnatural(dealercards) and isinsured == False:
            #time.sleep(0.5)
            drawnextdealercard(dealercards, color)
            #time.sleep(0.5)
            if isnatural(playercards):
                start, running = tie(color)
            else:
                start, running, score = playerlose(score, color)
        elif isnatural(playercards):
            drawnextdealercard(dealercards, color)
            start, running, score = dealerlose(score, color)
            score += 0.5
    return start, running, score, nextafterinsu
def insurance(running, action, color):
    lastkeypressed = -1
    showtext('insurance? [A-yes/D-no]', width*0.1, hight*0.5, color)
    while (lastkeypressed != pygame.K_a and lastkeypressed != pygame.K_d and running == True):
        if action == 0 or action == 1:
            lastkeypressed = pygame.K_a
        else:
            lastkeypressed = pygame.K_d

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
    showtext('insurance? [A-yes/D-no]', width*0.1, hight*0.5, (0, 0, 0))
    if lastkeypressed == pygame.K_a:
        return True, running
    return False, running
#%%
# Functions for nural net
def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_in', distribution='truncated_normal'))
def compute_avg_return(environment, policy, num_episodes):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]
def compute_avg_loss(iterator1):
    experience1, unused_info1 = next(iterator1)
    k = float(agent.train(experience1).loss)
    del unused_info1
    del experience1
    del iterator1
    return k
#%%
# Class for the game environment
class CardGameEnv(py_environment.PyEnvironment):

    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
                shape=(13,), dtype=np.int32, minimum=0, maximum=13, name='observation')
        self._episode_ended = False
        self._state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.creamcolor = [255, 253, 208]

        self.cardpack = []
        self.dealercards = []
        self.splitdealercard = []
        self.playercards = []
        self.lastpacksize = 0

        self.running = True
        self.start = True
        
        self.lastkeypressed = None
        self.firstmove = False
        self.score = 0
        self.lastscore = 0

        self.debugvar = True

        self.isstand = False
        self.cansplit = False
        self.issplit = 0
        self.splitcard = 0

        self.isdouble = 0
        self.candouble = False

        self.nextafterinsu = False
    
    def __start(self):
        self.isstand = False
        self.firstmove = True
        self.candouble = True
        self.lastscore = self.score
        self.playercards, self.dealercards, self.cardpack = initgame(self.playercards, self.dealercards, self.cardpack)
        while totalcards(self.playercards)[0] > 21:
            self.playercards, self.dealercards, self.cardpack = initgame(self.playercards, self.dealercards, self.cardpack)
        
        #if debugvar == True:
            #dealercards = [10, 1]
            #dealercards[0] = 1
            #playercards = [10, 1]
            #playercards[1] = playercards[0]
            #debugvar = False
        
        drawstarttext(self.playercards, self.score, self.creamcolor)
        
        # draws this if didnt lose at the start
        drawfirstdealercard(self.dealercards, self.creamcolor)

        # check split
        if (self.playercards[0] == self.playercards[1] or (max(totalcards(self.playercards)) == 20 and self.playercards[0] > 9)) and self.issplit == 0:
            self.cansplit = True
        elif self.issplit == 1 and self.firstmove == True:
            self.cardpack, self.playercards, self.dealercards, self.splitcard, self.splitdealercard, self.issplit = split(
                self.cardpack, self.playercards, self.dealercards, self.splitcard, self.splitdealercard, self.issplit, self.creamcolor)

        self._state = []
        for i in self.playercards:
            self._state.append(i)
        self._state.extend([0] * (6 - len(self._state)))
        self._state.append(self.dealercards[0])
        self._state.extend([0] * 6)
        #print(str(self._state))
    
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.__start()
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        if action < 0 or action > 3:
            raise ValueError('`action` should be 0 to 3.')

        self.nextafterinsu = False

        if self.start == True and self.running == True:
            self.start = False

            # check natural
            self.start, self.running, self.score, self.nextafterinsu = checknatural(
                self.dealercards, self.playercards, self.score, self.start, self.running, self.creamcolor, action)
            if self.running == False:
                #break
                pass
        
                
        self.lastpacksize = showpacksize(len(self.cardpack), self.lastpacksize, self.creamcolor)

        if self.running == True and self.start == False and self.nextafterinsu == False:
            if max(totalcards(self.playercards)) != 21:
                self.running, self.cardpack, self.playercards, self.isstand, self.cansplit, self.issplit, self.candouble, self.isdouble = asknextmove(
                    self.running, self.cardpack, self.playercards, self.isstand, self.cansplit, self.issplit, self.candouble, self.isdouble, self.creamcolor, action)
                if self.issplit == 2 and self.firstmove == True:
                    self.cardpack, self.playercards, self.dealercards, self.splitcard, self.splitdealercard, self.issplit = split(
                        self.cardpack, self.playercards, self.dealercards, self.splitcard, self.splitdealercard, self.issplit, self.creamcolor)
                self.start, self.running, self.score, self.isdouble = checkbust(
                    self.start, self.running, self.playercards, self.score, self.isdouble, self.creamcolor)
                if len(self.playercards) > 5 and self.start == False:
                    self.start, self.running, self.score = dealerlose(self.score, self.creamcolor)
            else:
                self.isstand = True

            self.firstmove = False

            if self.isstand == True and self.start == False:
                self.isstand = False
                self.start, self.running, self.cardpack, self.dealercards, self.score = dealermove(
                    self.start, self.running, self.cardpack, self.dealercards, self.score, self.creamcolor)
                if self.start == False and self.running == True:
                    if max(totalcards(self.dealercards)) > max(totalcards(self.playercards)):
                        self.start, self.running, self.score = playerlose(self.score, self.creamcolor)
                    elif max(totalcards(self.dealercards)) == max(totalcards(self.playercards)):
                        self.start, self.running = tie(self.creamcolor)
                    else:
                        self.start, self.running, self.score = dealerlose(self.score, self.creamcolor)
                if self.isdouble == 1:
                    self.isdouble = 0
                    self.score += self.score - self.lastscore

            self._state = []
            for i in self.playercards:
                self._state.append(i)
            self._state.extend([0] * (6 - len(self._state)))
            self._state.append(self.dealercards[0])
            self._state.extend([0] * 6)
            #print(str(self._state))
        
        # Make sure episodes don't go on forever.
        if self.start == True:
            self._episode_ended = True
            self._state = []
            for i in self.playercards:
                self._state.append(i)
            self._state.extend([0] * (6 - len(self._state)))
            for i in self.dealercards:
                self._state.append(i)
            self._state.extend([0] * (13 - len(self._state)))
            while len(self._state) > 13:
                self._state.pop(len(self._state) - 1)
            #print(str(self._state))
            #print(str(self.score) + ', ' + str(self.score - self.lastscore))
            return ts.termination(np.array(self._state, dtype=np.int32), self.score - self.lastscore)
        else:
            return ts.transition(np.array(self._state, dtype=np.int32), reward=0.0, discount=1.0)
#%%
pygame.init()

#Screen
hight = 720
width = 1280
screen = pygame.display.set_mode((width, hight))
pygame.display.set_caption('Blackjack')

font = pygame.font.Font(None, 46)

#%%

# tf init
tf.config.threading.set_intra_op_parallelism_threads(14)
tf.config.threading.set_inter_op_parallelism_threads(14)

strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)

train_env1 = CardGameEnv()
env1 = CardGameEnv()

#utils.validate_py_environment(env, episodes=20)

# environment init
train_env = tf_py_environment.TFPyEnvironment(train_env1)
eval_env = tf_py_environment.TFPyEnvironment(env1)

#Hyperparameter
collect_steps_per_iteration = 64  # @param {type:"integer"}
replay_buffer_capacity = 100000  # @param {type:"integer"}

batch_size = 1024  # @param {type:"integer"}
learning_rate = 1e-4  # @param {type:"number"}

fc_layer_params = (256, 256, 256, 256, 256, 256)
action_tensor_spec = tensor_spec.from_spec(eval_env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

#layers
dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(minval=-10, maxval=10),
    bias_initializer=tf.keras.initializers.RandomUniform(minval=-10, maxval=10))
q_net = sequential.Sequential(dense_layers + [q_values_layer])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

agent = dqn_agent.DdqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=tf.Variable(1))
agent.initialize()

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)

collect_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    agent.collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=collect_steps_per_iteration)

#error check
i = 2
while i > 0:
    i -= 1
    collect_driver.run()
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
    iterator = iter(dataset)
    experience, unused_info = next(iterator)
    agent.train(experience).loss


#checkpoint init
checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=agent)
manager = tf.train.CheckpointManager(
    checkpoint, "./data/tf", max_to_keep=1)
status = checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
    status.assert_consumed()  # Optional sanity checks.
else:
    print("nothing restored")

#%%
#loop init
agent.train = common.function(agent.train)

lastkeypressed = -1
returns_loss = []
returns_avg_return = []
log_interval = 1000
while True:
    collect_driver.run()
    #last_time_step = collect_driver.run()[0]
    #print(str(last_time_step[3].numpy()) + ', ' + str(float(last_time_step[2])))

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size,num_steps=2).prefetch(3)

    iterator = iter(dataset)

    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    if int(checkpoint.step) % 1000 == 0:
        manager.save()

    if int(checkpoint.step) % 100 == 0:
        print('[' + datetime.now().strftime("%H:%M:%S") + '] ' + 'step: ' + str(int(checkpoint.step)))

    if int(checkpoint.step) % log_interval == 0:
        #train_loss = compute_avg_loss(iterator)
        #del iterator
        train_avg_return = compute_avg_return(eval_env, agent.policy, 100)

        returns_loss.append(train_loss)
        returns_avg_return.append(train_avg_return)

        print('[' + datetime.now().strftime("%H:%M:%S") + '] ' + 'Iteration: {:} Average Loss: {:.2f} Average Return: {:.2f}'.format(
            int(checkpoint.save_counter), train_loss, train_avg_return))

    checkpoint.step.assign_add(1)

    #showtext('halt', width*0.8, hight*0.5, (100, 100, 100))
    #pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            lastkeypressed = event.key
    #showtext('halt', width*0.8, hight*0.5, (0, 0, 0))
    #pygame.display.update()
    if lastkeypressed == pygame.K_n:
        break
    
#save on exit
manager.save()

#draw graph loss-iterations
iterations = range(0, len(returns_loss), 1)
fig, axs = plt.subplots(2)
fig.suptitle('loss and avg return')
axs[0].plot(iterations, returns_loss)
axs[1].plot(iterations, returns_avg_return)
plt.show()