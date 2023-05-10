#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custamizable Catch Environment
By Thomas Moerland

AC with bootstrapping built bt R.Ma
"""

import matplotlib
matplotlib.use('TkAgg') #'Qt5Agg') # 'TkAgg'
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

ACTION_EFFECTS = (-1, 0, 1) # left, idle right. 
OBSERVATION_TYPES = ['pixel','vector']

class Catch():
    """
    Reinforcement learning environment where we need to move a paddle to catch balls that drop from the top of the screen. 
    
    -----------
    |     o   |
    |         |
    |         | 
    |         |
    |   _     |
    -----------
    
    o = ball
    _ = paddle
    
    
    State space: 
        - The width and height of the problem can be adjusted with the 'rows' and 'columns' argument upon initialization.
        - The observation space can either be a vector with xy-locations of paddle and lowest ball, 
        or a binary two-channel pixel array, with the paddle location in the first channel, and all balls in the second channel.
        This can be determined with the 'observation_type' argument upon initialization. 
        
    Action space: 
        - Each timestep the paddle can move left, right or stay idle.
        
    Reward function: 
        - When we catch a ball when it reaches the bottom row, we get a reward of +1. 
        - When we miss a ball that reaches the bottom row, we get a penalty of -1. 
        - All other situaties have a reward of 0.  
        
    Dynamcics function: 
        - Balls randomly drop from one of the possible positions on top of the screen.
        - The speed of dropping can be adjusted with the 'speed' parameter. 
        
    Termination: 
        - The task terminates when 1) we reach 'max_steps' total steps (to be set upon initialization), 
        or 2) we miss 'max_misses' total balls (to be set upon initialization). 
    
    """
  
    def __init__(self, rows: int = 7, columns: int = 7, speed: float = 1.0, 
                 max_steps: int = 250, max_misses: int = 10, 
                 observation_type: str = 'pixel', seed = None, 
                 ):
        """ Arguments: 
        rows: the number of rows in the environment grid.
        columns: number of columns in the environment grid.
        speed: speed of dropping new balls. At 1.0 (default), we drop a new ball whenever the last one drops from the bottom. 
        max_steps: number of steps after which the environment terminates.
        max_misses: number of missed balls after which the environment terminates (when this happens before 'max_steps' is reached).
        observation_type: type of observation, either 'vector' or 'pixel'. 
              - 'vector': observation is a vector of length 3:  [x_paddle,x_lowest_ball,y_lowest_ball]
              - 'pixel': observation is an array of size [rows x columns x 2], with one hot indicator for the paddle location in the first channel,
              and one-hot indicator for every present ball in the second channel. 
        seed: environment seed. 
        """
        if observation_type not in OBSERVATION_TYPES:
            raise ValueError('Invalid "observation_type". Needs to be in  {}'.format(OBSERVATION_TYPES))
        if speed <= 0.0:
            raise ValueError('Dropping "speed" should be larger than 0.0')
        
        # store arguments
        self._rng = np.random.RandomState(seed)
        self.rows = rows
        self.columns = columns
        self.speed = speed
        self.max_steps = max_steps
        self.max_misses = max_misses
        self.observation_type = observation_type
        
        # compute the drop interval 
        self.drop_interval = max(1,rows // speed) # compute the interval towards the next drop, can never drop below 1
        if speed != 1.0 and observation_type == 'vector': 
            print('Warning: You use speed > 1.0, which means there may be multiple balls in the screen at the same time.' +
                  'However, with observation_type = vector, only the xy location of *lowest* ball is visible to the agent' +
                  ' (to ensure a fixed length observation vector')
            
        # Initialize counter
        self.total_timesteps = None 
        self.fig = None
        self.action_space = spaces.Discrete(3,)
        if self.observation_type == 'vector':
            self.observation_space = spaces.Box(low=np.array((0,0,0)), high=np.array((self.columns, self.columns, self.rows)), dtype=int)
        elif self.observation_type == 'pixel':
            self.observation_space = spaces.Box(low=np.zeros((self.rows, self.columns, 2)), high=np.ones((self.rows, self.columns, 2)), dtype=int)

    def reset(self):
        ''' Reset the problem to empty board with paddle in the middle bottom and a first ball on a random location in the top row '''
        # reset all counters
        self.total_timesteps = 0
        self.total_reward = 0
        self.r = '-'
        self.missed_balls = 0
        self.time_till_next_drop = self.drop_interval
        self.terminal = False

        # initialized problem
        self.paddle_xy = [self.columns // 2, 0] # paddle in the bottom middle
        self.balls_xy = [] # empty the current balls
        self._drop_new_ball() # add the first ball
        s0 = self._get_state() # get first state
        return s0

    def step(self,a):
        ''' Forward the environment one step based on provided action a ''' 
        
        # Check whether step is even possible
        if self.total_timesteps is None: 
            ValueError("You need to reset() the environment before you can call step()")
        elif self.terminal: 
            ValueError("Environment has terminated, you need to call reset() first")
        
        # Move the paddle based on the chosen action
        self.paddle_xy[0] = np.clip(self.paddle_xy[0] + ACTION_EFFECTS[a],0,self.columns -1) 
        
        # Drop all balls one step down
        for ball in self.balls_xy: 
            ball[1] -= 1

        # Check whether lowest ball dropped from the bottom
        if len(self.balls_xy) > 0:         # there is a ball present
            if self.balls_xy[0][1] < 0:    # the lowest ball reached below the bottom
                del self.balls_xy[0]
        
        # Check whether we need to drop a new ball
        self.time_till_next_drop -= 1
        if self.time_till_next_drop == 0:
            self._drop_new_ball()
            self.time_till_next_drop = self.drop_interval 
            
        # Compute rewards
        if (len(self.balls_xy) == 0) or (self.balls_xy[0][1] != 0): # no ball present at bottom row
            r = 0.0 
        elif self.balls_xy[0][0] == self.paddle_xy[0]: # ball and paddle location match, caught a ball
            r = 1.0 
        else: # missed the ball
            r = -1.0
            self.missed_balls += 1
        
        # Compute termination
        self.total_timesteps += 1
        if (self.total_timesteps == self.max_steps) or (self.missed_balls == self.max_misses):
            self.terminal = True
        else:
            self.terminal = False
        
        self.r = r
        self.total_reward += r
        return self._get_state(), r, self.terminal
    
    def render(self,step_pause=0.3):
        ''' Render the current environment situation '''
        if self.total_timesteps is None: 
            ValueError("You need to reset() the environment before you render it")
        
        # In first call initialize figure
        if self.fig == None:
            self._initialize_plot()

        # Set all colors to white
        for x in range(self.columns):
            for y in range(self.rows):
                if self.paddle_xy == [x,y]: # hit the agent location
                    if [x,y] in self.balls_xy: # agent caught a ball
                        self.patches[x][y].set_color('g')
                    else: 
                        self.patches[x][y].set_color('y')
                elif [x,y] in self.balls_xy: # hit a ball location without agent
                    if y == 0: # missed the ball
                        self.patches[x][y].set_color('r')
                    else: # just a ball
                        self.patches[x][y].set_color('w')
                else: # empty spot
                    self.patches[x][y].set_color('k')
        #plt.axis('off')
        
        self.label.set_text('Reward:  {:<5}            Total reward:  {:<5}     \nTotal misses: {:>2}/{:<2}     Timestep: {:>3}/{:<3}'.format(
            self.r,self.total_reward,self.missed_balls,self.max_misses,self.total_timesteps,self.max_steps))

        # Draw figure
        plt.pause(step_pause)                  

        
    def _initialize_plot(self):
        ''' initializes the catch environment figure ''' 
        self.fig,self.ax = plt.subplots()
        self.fig.set_figheight(self.rows)
        self.fig.set_figwidth(self.columns)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim([0,self.columns])
        self.ax.set_ylim([0,self.rows]) 
        self.ax.axes.xaxis.set_visible(False)
        self.ax.axes.yaxis.set_visible(False)

        self.patches = [ [[] for x in range(self.rows)] for y in range(self.columns)]
        for x in range(self.columns):
            for y in range(self.rows):
                self.patches[x][y] = Rectangle((x, y),1,1, linewidth=0.0, color = 'k') 
                self.ax.add_patch(self.patches[x][y]) 
                
        self.label = self.ax.text(0.01,self.rows + 0.2,'', fontsize=20, c='k')
        
    def _drop_new_ball(self):
        ''' drops a new ball from the top ''' 
        self.balls_xy.append([self._rng.randint(self.columns),self.rows-1])#0])
            
    def _get_state(self):
        ''' Returns the current agent observation '''
        if self.observation_type == 'vector': 
            if len(self.balls_xy) > 0: # balls present
                s = np.append(self.paddle_xy[0],self.balls_xy[0]).astype('float32') # paddle xy and ball xy 
            else: 
                s = np.append(self.paddle_xy[0],[-1,-1]).astype('float32') # no balls, impute (-1,-1) in state for no ball present
        elif self.observation_type == 'pixel':
            s = np.zeros((self.columns,self.rows, 2), dtype=np.float32)
            s[self.paddle_xy[0], self.paddle_xy[1], 0] = 1.0 # set paddle indicator in first slice
            for ball in self.balls_xy:
                s[ball[0], ball[1], 1] = 1.0 # set ball indicator(s) in second slice
        else:
            raise ValueError('observation_type not recognized, needs to be in {}'.format(OBSERVATION_TYPES))
        return s

class Hyperparameters():
    def __init__(self):
        self.entro_param = 0.01
        self.gamma = 0.99
        self.rows = 7
        self.columns = 7
        self.speed = 1.0
        self.max_steps = 250
        self.max_misses = 10
        self.observation_type = 'pixel'  # 'vector'
        self.seed = None
        self.hidden_size = 64
        self.num_episodes = 2000


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        hyp = Hyperparameters()
        state_size = hyp.rows * hyp.columns * 2
        action_size = 3
        hidden_size = 64
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        hyp = Hyperparameters()
        state_size = hyp.rows * hyp.columns * 2
        hidden_size = 64
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def actor_critic_with_bootstrapping():
    hyp = Hyperparameters()
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device is potentially faster")
    actor_model = Actor().to(device)
    critic_model = Critic().to(device)
    print("The actor model is: ", actor_model)
    print("The critic model is: ", critic_model)

    # Initialize environment and Q-array
    env = Catch(rows=hyp.rows, columns=hyp.columns, speed=hyp.speed, max_steps=hyp.max_steps,
                max_misses=hyp.max_misses, observation_type=hyp.observation_type, seed=hyp.seed)

    s = env.reset()
    step_pause = 0.3  # the pause between each plot
    #env.render(step_pause)

    actor = Actor()
    critic = Critic()
    optimizer_actor = optim.Adam(actor.parameters(), lr=0.001)
    optimizer_critic = optim.Adam(critic.parameters(), lr=0.001)

    scores = []
    avg_scores = []
    actor_losses = []
    critic_losses = []
    num_episodes = hyp.num_episodes
    gamma = hyp.gamma

    for i_episode in range(num_episodes):
        state = env.reset()
        done = False
        score = 0
        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            action_probs = actor(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            next_state, reward, done = env.step(action.item())
            score += reward
            next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)
            value = critic(state_tensor)
            next_value = critic(next_state_tensor)
            td_error = reward + gamma * next_value

            # Update critic
            loss_critic =  F.mse_loss(value, td_error)
            optimizer_critic.zero_grad()
            loss_critic.backward()
            optimizer_critic.step()

            # Update actor
            log_prob = dist.log_prob(action)
            entropy = dist.entropy().mean()
            loss_actor = -log_prob * td_error.item() - hyp.entro_param * entropy
            optimizer_actor.zero_grad()
            loss_actor.backward()
            optimizer_actor.step()
            if done:
                state = env.reset()
                scores.append(score)
                avg_scores.append(np.mean(scores[-50:]))
                actor_losses.append(loss_actor.item())
                critic_losses.append(loss_critic.item())
            else:
                state = next_state
        print('Episode {}\tEpisode Score: {:.2f}'.format(i_episode, scores[i_episode]))

    plot_training_progress(scores, avg_scores, actor_losses, critic_losses)


def plot_training_progress(scores, avg_scores, actor_losses, critic_losses):
    """Plot the training progress of the actor-critic algorithm."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Plot the raw scores
    axs[0, 0].plot(scores)
    axs[0, 0].set_title('Raw scores')

    # Plot the rolling average of the scores
    axs[0, 1].plot(avg_scores)
    axs[0, 1].set_title('Rolling average scores')

    # Plot the actor loss over time
    axs[1, 0].plot(actor_losses)
    axs[1, 0].set_title('Actor loss')

    # Plot the critic loss over time
    axs[1, 1].plot(critic_losses)
    axs[1, 1].set_title('Critic loss')

    # Add axis labels and titles
    for ax in axs.flat:
        ax.set(xlabel='Episode', ylabel='Value')
    fig.suptitle('Training Progress of Actor-Critic with only bootstrapping', fontsize=16)

    fontsize = 12
    for ax in axs.flat:
        ax.tick_params(labelsize=fontsize)
        ax.set_title(ax.get_title(), fontsize=fontsize)
        ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
    fig.tight_layout()

    # Show the plot
    plt.show()


if __name__ == '__main__':

    actor_critic_with_bootstrapping()
        
