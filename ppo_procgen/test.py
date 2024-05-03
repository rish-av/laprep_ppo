from typing import Any
import jax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
import gym

class q_net(nn.Module):

    action_dim: int
    def __call__(self, x: jnp.ndarray):

        x = nn.Conv(32, 3 , padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(32, 3, padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(16, 3, padding='SAME')(x)
        x = nn.relu(x)

        x = x.reshape(x.shape[0],-1)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        q_val = nn.Dense(self.action_dim)(x)

        return q_val

class ReplayBuffer():

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def append(self, state, action, reward, next_state, done):

        self.buffer.append([state, action, reward, next_state, done])
    
    def get_samples(self):

        self.buffer = np.random.permutation(self.buffer)
        idx = np.random.randint(0, len(self.buffer))
        return self.buffer[idx]
    
@jax.jit
def update(q_state:TrainState, 
           params, 
           obs,
           action,
           reward,
           next_obs,
           done,
           target_params):
    

    q_pred = q_state.apply_fn({'params':params}, obs)
    q_pred = q_pred[jnp.arange(q_pred.shape[0], action.squeeze())]
    q_next_target = jnp.max(q_state.apply_fn({'params':target_params}, next_obs), axis=-1)
    next_q_val = reward + (1-done)*0.99*q_next_target
    loss = ((next_q_val - q_pred)**2).mean()
    return loss, q_pred

    
     

def make_env():

    #make a vector env
    env = None
    return env




def train():

    env = gym.make('CartPole-v1')
    net = q_net(env.action_space.shape)


    buffer_size = 50000
    rb = ReplayBuffer(buffer_size)

    obs, _ = env.reset()


    #take an action given obs
    


    model = q_net()
    state = TrainState.create()
    epsilon = 0.0001

