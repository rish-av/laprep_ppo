from typing import Any, Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
from jax import lax
from tensorflow_probability.substrates import jax as tfp

import jax

tfd = tfp.distributions
tfb = tfp.bijectors


def default_conv_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.xavier_uniform()

def default_mlp_init(scale: Optional[float] = 0.01):
    return nn.initializers.orthogonal(scale)

def default_logits_init(scale: Optional[float] = 0.01):
    return nn.initializers.orthogonal(scale)


class ResidualBlock(nn.Module):
    """Residual block."""
    num_channels: int
    prefix: str

    @nn.compact
    def __call__(self, x):
        # Conv branch
        y = nn.relu(x)
        y = nn.Conv(self.num_channels,
                    kernel_size=[3, 3],
                    strides=(1, 1),
                    padding='SAME',
                    kernel_init=default_conv_init(),
                    name=self.prefix + '/conv2d_1')(y)
        y = nn.relu(y)
        y = nn.Conv(self.num_channels,
                    kernel_size=[3, 3],
                    strides=(1, 1),
                    padding='SAME',
                    kernel_init=default_conv_init(),
                    name=self.prefix + '/conv2d_2')(y)

        return y + x



class CNN_feat(nn.Module):
    prefix: str

    @nn.compact
    def __call__(self,x):
        out = x
        for i, (num_channels, num_blocks) in enumerate([(16, 2), (16, 2),
                                                        (16, 2)]):
            conv = nn.Conv(num_channels,
                           kernel_size=[3, 3],
                           strides=(1, 1),
                           padding='SAME',
                           kernel_init=default_conv_init(),
                           name=self.prefix + '/conv2d_%d' % i)
            out = conv(out)

            out = nn.max_pool(out,
                              window_shape=(3, 3),
                              strides=(2, 2),
                              padding='SAME')
            for j in range(num_blocks):
                block = ResidualBlock(num_channels,
                                      prefix='residual_{}_{}'.format(i, j))
                out = block(out)
        return out



class Impala(nn.Module):
    """IMPALA architecture."""
    prefix: str

    @nn.compact
    def __call__(self, x):
        out = x
        for i, (num_channels, num_blocks) in enumerate([(16, 2), (32, 2),
                                                        (32, 2)]):
            conv = nn.Conv(num_channels,
                           kernel_size=[3, 3],
                           strides=(1, 1),
                           padding='SAME',
                           kernel_init=default_conv_init(),
                           name=self.prefix + '/conv2d_%d' % i)
            out = conv(out)

            out = nn.max_pool(out,
                              window_shape=(3, 3),
                              strides=(2, 2),
                              padding='SAME')
            for j in range(num_blocks):
                block = ResidualBlock(num_channels,
                                      prefix='residual_{}_{}'.format(i, j))
                out = block(out)

        out = out.reshape(out.shape[0], -1)
        out = nn.relu(out)
        out = nn.Dense(256, kernel_init=default_mlp_init(), name=self.prefix + '/representation')(out)
        out = nn.relu(out)
        return out


class TwinHeadModel(nn.Module):
    """Critic+Actor for PPO."""
    action_dim: int
    prefix_critic: str = "critic"
    prefix_actor: str = "policy"

    @nn.compact
    def __call__(self, x):
        z = Impala(prefix='shared_encoder')(x)
        # Linear critic
        v = nn.Dense(1, kernel_init=default_mlp_init(), name=self.prefix_critic + '/fc_v')(z)

        logits = nn.Dense(self.action_dim,
                        kernel_init=default_logits_init(),
                        name=self.prefix_actor + '/fc_pi')(z)

        pi = tfd.Categorical(logits=logits)
        
        return v, pi


class LapNetwork(nn.Module):
    out_channels: int
    embedding_dim: int

    @nn.compact
    def __call__(self, data):
        x = data
        x = Impala(prefix='lap_network')(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        lapembeddings = nn.Dense(self.embedding_dim)(x)
        return lapembeddings

class DiffusionNetwork(nn.Module):
    action_dim: int
    diff_power: int
    inverse: bool

    prefix_critic: str = "critic"
    prefix_actor: str = "policy"

    @nn.compact
    def __call__(self, lap_embeddings, eigenvalues=None, train=False):
        h = lax.stop_gradient(lap_embeddings)
        h = nn.BatchNorm(use_running_average=not train)(h)
        h = nn.Dense(128)(h)
        h = nn.relu(h)
        h = nn.Dense(64)(h)
        h = nn.relu(h)
        
        v = nn.Dense(1, kernel_init=default_mlp_init(), name=self.prefix_critic + '/fc_v')(h)

        logits = nn.Dense(self.action_dim,
                        kernel_init=default_logits_init(),
                        name=self.prefix_actor + '/fc_pi')(h)

        pi = tfd.Categorical(logits=logits)
        
        return v, pi



class RegularizerModel(nn.Module):

    embedding_dim:int
    action_dim:int
    prefix_critic: str = "critic"
    prefix_actor: str = "policy"

    @nn.compact
    def __call__(self,x):

        #cnn
        # x = nn.Conv(32, kernel_size=(3,3), padding='SAME')(x)
        # x = nn.relu(x)
        # x = nn.Conv(32, kernel_size=(3,3), padding='SAME')(x)
        # x = nn.relu(x)
        # x = nn.Conv(32, kernel_size=(3,3), padding='SAME')(x)
        # x = nn.relu(x)

        # u = x.reshape(x.shape[0], -1)
        # u = nn.Dense(128)(u)
        # u = nn.relu(u)
        # u = nn.Dense(64)(u)
        # lapembeddings = nn.Dense(self.embedding_dim)(u)



        #ppo
        # x = x.reshape(x.shape[0], -1)
        # x = nn.relu(x)
        # x = nn.Dense(256, kernel_init=default_mlp_init(), name= 'ppo'+ '/representation')(x)
        # x = nn.relu(x)
        # x = x/255.
        z = Impala(prefix='shared_encoder')(x)
        v = nn.Dense(1, kernel_init=default_mlp_init(), name=self.prefix_critic + '/fc_v')(z)
        logits = nn.Dense(self.action_dim,
                        kernel_init=default_logits_init(),
                        name=self.prefix_actor + '/fc_pi')(z)

        pi = tfd.Categorical(logits=logits)
        return v, pi

class RegularizerModelTest(nn.Module):

    embedding_dim:int
    action_dim:int
    prefix_critic: str = "critic"
    prefix_actor: str = "policy"

    @nn.compact
    def __call__(self,x):

        #cnn
        # x = nn.Conv(32, kernel_size=(3,3), padding='SAME')(x)
        # x = nn.relu(x)
        # x = nn.Conv(32, kernel_size=(3,3), padding='SAME')(x)
        # x = nn.relu(x)
        # x = nn.Conv(32, kernel_size=(3,3), padding='SAME')(x)
        # x = nn.relu(x)

        z = CNN_feat(prefix='shared_encoder')(x)

        u = z.reshape(z.shape[0], -1)
        u = nn.Dense(128)(u)
        u = nn.relu(u)
        u = nn.Dense(64)(u)
        u = nn.relu(u)
        lapembeddings = nn.Dense(self.embedding_dim)(u)


        z = z.reshape(z.shape[0], -1)
        z = nn.relu(z)
        z = nn.Dense(256, kernel_init=default_mlp_init(), name='ppo' + '/representation')(z)
        z = nn.relu(z)

        v = nn.Dense(1, kernel_init=default_mlp_init(), name=self.prefix_critic + '/fc_v')(z)

        logits = nn.Dense(self.action_dim,
                        kernel_init=default_logits_init(),
                        name=self.prefix_actor + '/fc_pi')(z)

        pi = tfd.Categorical(logits=logits)

        return v, pi, lapembeddings
    

class FeatureFusion(nn.Module):


    action_dim:int
    embedding_dim: int
    diff_power:int
    inverse: bool = False

    @nn.compact
    def __call__(self, x, train=False):
        #image 
        z = CNN_feat('image_encoder')(x)
        i = z.reshape(z.shape[0], -1)
        i = nn.relu(i)
        i = nn.Dense(256, kernel_init=default_mlp_init(), name='ppo' + '/representation')(i)
        i = nn.relu(i)


        #image value and logits
        v_img = nn.Dense(1, kernel_init=default_mlp_init(), name='img' + '/fc_v')(i)
        logits_img = nn.Dense(self.action_dim,
                        kernel_init=default_logits_init(),
                        name='img' + '/fc_pi')(i)

        #lap_net
        u = Impala(prefix='lap_network')(x)
        u = nn.Dense(128)(u)
        u = nn.relu(u)
        u = nn.Dense(64)(u)
        u = nn.relu(u)
        lap_embeddings = nn.Dense(self.embedding_dim)(u)
        u = jax.lax.stop_gradient(lap_embeddings)
        # eigenvalues = jax.lax.stop_gradient(eigenvalues)
        # if not self.inverse:
        #     eigenvalues = jnp.power(eigenvalues, self.diff_power)
        # else:
        #     eigenvalues = jnp.power(eigenvalues, self.diff_power)
        #     eigenvalues = jnp.nan_to_num(1 / jnp.sqrt(eigenvalues), nan=0, neginf=0)

        # u = jnp.multiply(eigenvalues, u)
        u = nn.BatchNorm(use_running_average=not train)(u)
        u = nn.Dense(128)(u)
        u = nn.relu(u)
        e = nn.Dense(64)(u)
        e = nn.relu(e)


        #lap value and logits
        v_lap = nn.Dense(1, kernel_init=default_mlp_init(), name='lap' + '/fc_v')(e)
        logits_lap = nn.Dense(self.action_dim)(e)


        mha_1 = nn.attention.SelfAttention(num_heads=1, deterministic=True)
        mha_2 = nn.attention.SelfAttention(num_heads=1, deterministic=True)
        
        vcombined = jnp.concatenate((jnp.expand_dims(v_lap, 1), jnp.expand_dims(v_img, 1)), axis=1)
        vcombined = mha_2(vcombined)
        vcombined = jnp.sum(vcombined, axis=1, keepdims=False)

        q = jnp.concatenate((jnp.expand_dims(logits_lap, 1), jnp.expand_dims(logits_img, 1)), axis=1)  # B x S X D
        q = mha_1(q)
        logits = jnp.sum(q, axis=1, keepdims=False)

        pi = tfd.Categorical(logits=logits)

        return vcombined, pi, lap_embeddings
    
class AlloNet(nn.Module):

    embedding_dim:int
    action_dim:int
    diff_power: int
    inverse: bool = False


    @nn.compact
    def __call__(self, x, eigenvalues, train=False):

        u = Impala(prefix='lap_network')(x)
        u = nn.Dense(128)(u)
        u = nn.relu(u)
        u = nn.Dense(64)(u)
        u = nn.relu(u)
        lap_embeddings = nn.Dense(self.embedding_dim)(u)
        u = jax.lax.stop_gradient(lap_embeddings)
        
        eigenvalues = jax.lax.stop_gradient(eigenvalues)
        if not self.inverse:
            eigenvalues = jnp.power(eigenvalues, self.diff_power)
        else:
            eigenvalues = jnp.power(eigenvalues, self.diff_power)
            eigenvalues = jnp.nan_to_num(1 / jnp.sqrt(eigenvalues), nan=0, neginf=0)
        u = jnp.multiply(eigenvalues, u)

        u = nn.BatchNorm(use_running_average=not train)(u)
        u = nn.Dense(128)(u)
        u = nn.relu(u)
        e = nn.Dense(64)(u)
        e = nn.relu(e)


        #lap value and logits
        v_lap = nn.Dense(1, kernel_init=default_mlp_init(), name='lap' + '/fc_v')(e)
        logits_lap = nn.Dense(self.action_dim)(e)


        pi = tfd.Categorical(logits=logits_lap)

        return v_lap, pi, lap_embeddings