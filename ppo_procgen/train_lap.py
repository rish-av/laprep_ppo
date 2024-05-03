import os
from typing import Any
import jax
import sys 
import optax
import wandb
import numpy as np
import jax.numpy as jnp
import gym
from flax.training.train_state import TrainState
from procgen import ProcgenEnv
import orbax.checkpoint as ocp
from etils import epath
import chex
from vec_env import ProcgenVecEnvCustom
from models import AlloNet as allo
from absl import app, flags
from typing import Callable
import tqdm
from collections import deque
from buffer_lap import BatchReg
from flax.training import checkpoints

from algo_lap import update_regularizer, get_transition_reg, select_action_reg

def safe_mean(x):
    return np.nan if len(x) == 0 else np.mean(x)

FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "starpilot", "Env name")
flags.DEFINE_float("barrier_val", 1.0, "Barrier Value")
flags.DEFINE_float("lap_lr", 1e-4, "laplacian learning rate")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("num_envs", 32, "Num of Procgen envs.")
flags.DEFINE_integer("train_steps", 25_000_000, "Number of train frames.")
# PPO
flags.DEFINE_float("max_grad_norm", 0.5, "Max grad norm")
flags.DEFINE_float("gamma", 0.999, "Gamma")
flags.DEFINE_integer("n_steps", 256, "GAE n-steps")
flags.DEFINE_integer("n_minibatch", 8, "Number of PPO minibatches")
flags.DEFINE_float("lr", 5e-4, "PPO learning rate")
flags.DEFINE_integer("epoch_ppo", 3, "Number of PPO epochs on a single batch")
flags.DEFINE_float("clip_eps", 0.2, "Clipping range")
flags.DEFINE_float("gae_lambda", 0.95, "GAE lambda")
flags.DEFINE_float("entropy_coeff", 0.01, "Entropy loss coefficient")
flags.DEFINE_float("critic_coeff", 0.5, "Value loss coefficient")
# Logging
flags.DEFINE_integer("checkpoint_interval", 999424, "Chcekpoint frequency (about 1M)")
flags.DEFINE_string("model_dir", "model_weights", "Model weights dir")
flags.DEFINE_string("run_id", "jax_ppo",
                    "Run ID. Change that to change W&B name")
flags.DEFINE_boolean("disable_wandb", False, "Disable W&B logging")
flags.DEFINE_string("method_name","allo", "Name of the method")
flags.DEFINE_integer("diff_power", 0, "eigenvalue scaling")


FLAGS(sys.argv)


class Lap_TrainState(TrainState):
    betas: Any
    barrier_coeffs: Any
    errors: Any
    quadratic_errors: Any
    batch_stats: Any


class PPO_TrainState(TrainState):
    batch_stats: Any

@chex.dataclass(frozen=True)
class TimeStep:
    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def main():
    env = ProcgenVecEnvCustom(FLAGS.env_name,
                            num_levels=200,
                            mode='easy',
                            start_level=0,
                            paint_vel_info=False,
                            num_envs=FLAGS.num_envs,
                            normalize_rewards=True,
                            use_backgrounds=False)

    env_test_ID = ProcgenVecEnvCustom(FLAGS.env_name,
                                    num_levels=200,
                                    mode='easy',
                                    start_level=0,
                                    paint_vel_info=False,
                                    num_envs=FLAGS.num_envs,
                                    normalize_rewards=False,
                                    use_backgrounds=False)
    env_test_OOD = ProcgenVecEnvCustom(FLAGS.env_name,
                                    num_levels=0,
                                    mode='easy',
                                    start_level=0,
                                    paint_vel_info=True,
                                    num_envs=FLAGS.num_envs,
                                    normalize_rewards=False,
                                    use_backgrounds=False)


    wandb.init(project='lap_reg_ppo', entity='revisit-spectral-rl', config=FLAGS, name='allo_ppo_{}_{}'.format(FLAGS.env_name, FLAGS.seed), 
            sync_tensorboard=True, group=str(FLAGS.env_name + '_' + FLAGS.method_name))

    np.random.seed(FLAGS.seed)
    key = jax.random.PRNGKey(FLAGS.seed)


    n_actions = env_test_ID.action_space.n
    lap_embed_dim = 128 #make this an argument

    model = allo(embedding_dim=lap_embed_dim, action_dim=n_actions, diff_power=0, inverse=False)
    params_model = model.init(key, jnp.zeros((1, 64, 64, 3)), jnp.ones((1, lap_embed_dim)), train=False)



    train_state = Lap_TrainState.create(apply_fn=model.apply, params=params_model,
                                        betas=jnp.zeros((lap_embed_dim, lap_embed_dim)),
                                        barrier_coeffs=jnp.tril(jnp.ones((1, 1)), k=0) * FLAGS.barrier_val,
                                        errors=jnp.zeros((lap_embed_dim, lap_embed_dim)),
                                        quadratic_errors=jnp.zeros((1, 1)),
                                        tx=optax.adam(learning_rate=FLAGS.lap_lr, eps=1.5e-4),
                                        batch_stats=params_model['batch_stats'])
    

    batch = BatchReg(
            discount=FLAGS.gamma,
            gae_lambda=FLAGS.gae_lambda,
            n_steps=FLAGS.n_steps+1,
            num_envs=FLAGS.num_envs,
            state_space=env.observation_space,
        )


    

    state = env.reset()
    state_id = env_test_ID.reset()
    state_ood = env_test_OOD.reset()

    epinfo_buf_id = deque(maxlen=100)
    epinfo_buf_ood = deque(maxlen=100)

    model_dir = os.path.join(FLAGS.model_dir, FLAGS.env_name, FLAGS.method_name)

    for step in tqdm.tqdm(range(1, int(FLAGS.train_steps // FLAGS.num_envs + 1))):

        #ppo training
        
        train_state, state, batch, key = get_transition_reg(train_state, env, state, batch, key, train=False)

        action_id, _, _, key = select_action_reg(train_state, state_id.astype(jnp.float32) / 255., key, sample=True, train=False)
        state_id, _, _, infos_id = env_test_ID.step(action_id)

        action_ood, _, _, key = select_action_reg(train_state, state_ood.astype(jnp.float32) / 255., key, sample=True, train =False)
        state_ood, _, _, infos_ood = env_test_OOD.step(action_ood)


        for info in infos_id:
            maybe_epinfo = info.get('episode')
            if maybe_epinfo:
                epinfo_buf_id.append(maybe_epinfo)

        for info in infos_ood:
            maybe_epinfo = info.get('episode')
            if maybe_epinfo:
                epinfo_buf_ood.append(maybe_epinfo)

        if step % (FLAGS.n_steps + 1) == 0:
            
            metric_dict, train_state, key = update_regularizer(
                                                    train_state,
                                                    batch.get(),
                                                    FLAGS.num_envs,
                                                    FLAGS.n_steps,
                                                    FLAGS.n_minibatch,
                                                    FLAGS.epoch_ppo,
                                                    FLAGS.clip_eps,
                                                    FLAGS.entropy_coeff,
                                                    FLAGS.critic_coeff,
                                                    key)
            batch.reset()
            renamed_dict = {}
            for k,v in metric_dict.items():
                renamed_dict["%s/%s"%(FLAGS.env_name,k)] = v

            for k,v in renamed_dict.items():
                    wandb.log({
                        k:v,
                        "step":FLAGS.num_envs * step
                    })

            wandb.log({
                "%s/ep_return_200" % (FLAGS.env_name): safe_mean([info['r'] for info in epinfo_buf_id]),
                "step": FLAGS.num_envs * step
            })
            wandb.log({
                "%s/ep_return_all" % (FLAGS.env_name): safe_mean([info['r'] for info in epinfo_buf_ood]),
                "step": FLAGS.num_envs * step
            })
            print('Eprew: %.3f'%safe_mean([info['r'] for info in epinfo_buf_id]))
            # print("Logits L1 norm: %.3f" % jnp.linalg.norm(ppo_state.params['policy/fc_pi']['kernel']) )

        if ( step * FLAGS.num_envs ) % FLAGS.checkpoint_interval == 0:
            print('Saving model weights [skipping for now]')
            checkpoints.save_checkpoint(os.path.abspath('./%s'%model_dir), train_state, step * FLAGS.num_envs)

main()