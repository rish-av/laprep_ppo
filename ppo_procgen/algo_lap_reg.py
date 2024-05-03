from collections import defaultdict
from functools import partial
from typing import Any, Callable, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from jax.random import PRNGKey
import flax

from algo_lap import flatten_dims
from collections import deque


def discounted_sampling(ranges, discount):
    assert np.min(ranges) >= 1
    assert discount >= 0 and discount <= 1
    seeds = np.random.uniform(size=ranges.shape)
    if discount == 0:
        samples = np.zeros_like(seeds, dtype=np.int64)
    elif discount == 1:
        samples = np.floor(seeds * ranges).astype(np.int64)
    else:
        samples = (np.log(1 - (1 - np.power(discount, ranges)) * seeds) 
                / np.log(discount))
        samples = np.floor(samples).astype(np.int64)
    return samples

def proper_laplacian(curr_phi, next_phi, neg_phi_u, barrier_coeffs, betas, error_estimates,
                     use_dual=True, ind=0, save_metrics=True):
    batch_size, n_dim = curr_phi.shape
    coefficient_vector = jnp.ones(n_dim)
    neg_phi_v = curr_phi
    lap_dim = n_dim
    dual_loss = 0
    pos_dot = neg_dot = neg_norm = None

    def compute_graph_drawing_loss(curr_phi, next_phi):
        '''Compute reprensetation distances between start and end states'''

        graph_induced_norms = ((curr_phi - next_phi) ** 2).mean(0)
        loss = graph_induced_norms.dot(coefficient_vector)

        return loss

    def compute_orthogonality_error_matrix(neg_phi_u, neg_phi_v):
        n = neg_phi_u.shape[0]

        inner_product_matrix_1 = jnp.einsum('ij,ik->jk', neg_phi_u, jax.lax.stop_gradient(neg_phi_u)) / n
        inner_product_matrix_2 = jnp.einsum('ij,ik->jk', neg_phi_v, jax.lax.stop_gradient(neg_phi_v)) / n

        error_matrix_1 = jnp.tril(inner_product_matrix_1 - jnp.eye(lap_dim))
        error_matrix_2 = jnp.tril(inner_product_matrix_2 - jnp.eye(lap_dim))
        error_matrix = 0.5 * (error_matrix_1 + error_matrix_2)
        quadratic_error_matrix = error_matrix_1 * error_matrix_2

        new_error_matrix_dict = {'errors': error_matrix, 'quadratic_errors': quadratic_error_matrix}

        return new_error_matrix_dict

    def compute_dual_loss(betas, error_matrix):
        return (jax.lax.stop_gradient(betas) * error_matrix).sum()

    def compute_barrier_loss(barrier, quadratic_error_matrix):
        quadratic_error_term = quadratic_error_matrix.sum()
        return jax.lax.stop_gradient(barrier[0, 0]) * quadratic_error_term, quadratic_error_term

    def update_error_estimates(errors):
        updates = {}
        for error_type in ['errors', 'quadratic_errors']:
            if not use_dual and error_type == 'errors':
                continue
            # Get old error estimates
            old = error_estimates[error_type]
            norm_old = jnp.linalg.norm(old)

            # Set update rate to 1 in the first iteration
            init_coeff = jnp.isclose(norm_old, 0.0, rtol=1e-10, atol=1e-13)
            non_init_update_rate = 1. if error_type == 'errors' else 0.1
            update_rate = init_coeff + (1 - init_coeff) * non_init_update_rate

            # Update error estimates
            update = old + update_rate * (errors[error_type] - old)  # The first update might be too large
            updates[error_type] = update

        return updates

    # Positive Constraint
    pos_loss = compute_graph_drawing_loss(curr_phi, next_phi)

    # Orthogonal Constraint
    error_matrix_dict = compute_orthogonality_error_matrix(neg_phi_u, neg_phi_v)
    barrier_loss, quadratic_error = compute_barrier_loss(barrier_coeffs, error_matrix_dict['quadratic_errors'])
    if use_dual:
        dual_loss = compute_dual_loss(betas, error_matrix_dict['errors'])

    # Total loss
    loss = pos_loss + dual_loss + barrier_loss

    # Update error estimates
    new_error_estimates = update_error_estimates(error_matrix_dict)

    # Monitor metrics
    if save_metrics:
        pos_dot = jnp.sum(next_phi[:, ind:] * curr_phi[:, ind:], axis=1).mean()
        neg_dot = jnp.sum(neg_phi_u[:, ind:] * neg_phi_v[:, ind:], axis=1).mean()
        neg_norm = (jnp.linalg.norm(neg_phi_u[:, ind:], ord=2, axis=1).mean() + jnp.linalg.norm(neg_phi_v[:, ind:], ord=2, axis=1).mean())/2

    return (loss, new_error_estimates), (pos_dot, quadratic_error, neg_dot, neg_norm, pos_loss)

def update_duals(error_estimates, betas, dual_lr=1e-4, min_dual=-500, max_dual=500):
    '''
      Update dual variables using some approximation
      of the gradient of the lagrangian.
    '''
    error_matrix = error_estimates['errors']
    dual_variables = betas
    updates = jnp.tril(error_matrix)

    # Calculate updated duals
    # dual_lr = 0.001  # old_val
    updated_duals = dual_variables + dual_lr * updates

    # Clip duals to be in the range [min_duals, max_duals]
    updated_duals = jnp.clip(
        updated_duals,
        a_min=min_dual,
        a_max=max_dual,
    )  # TODO: Cliping is probably not the best way to handle this

    # Update params, making sure that the duals are lower triangular
    betas = jnp.tril(updated_duals)

    return betas



def update_barrier_coefficients(barrier_coefficients, error_estimates, lr_barrier_coeffs=1.0,
                                min_barrier_coeff=0.0, max_barrier_coeff=10000.0):
    '''
        Update barrier coefficients using some approximation
        of the barrier gradient in the modified lagrangian.
    '''
    quadratic_error_matrix = error_estimates['quadratic_errors']
    updates = jnp.clip(quadratic_error_matrix, a_min=0, a_max=None).mean()

    # Calculate updated coefficients
    # barrier_lr = 0.01  # old value
    updated_barrier_coefficients = barrier_coefficients + lr_barrier_coeffs * updates

    # Clip coefficients to be in the range [min_barrier_coefs, max_barrier_coefs]
    updated_barrier_coefficients = jnp.clip(
        updated_barrier_coefficients,
        a_min=min_barrier_coeff,
        a_max=max_barrier_coeff,
    )  # TODO: Cliping is probably not the best way to handle this

    return updated_barrier_coefficients



def combined_loss(
    params_model: flax.core.frozen_dict.FrozenDict,
    apply_fn: Callable[..., Any],
    state: jnp.ndarray,
    next_state,
    neg_state,
    target: jnp.ndarray,
    value_old: jnp.ndarray,
    log_pi_old: jnp.ndarray,
    gae: jnp.ndarray,
    action: jnp.ndarray,
    errors,
    quadratic_errors,
    barrier_coeffs,
    betas,
    clip_eps: float,
    critic_coeff: float,
    entropy_coeff: float):


    #lap loss
    error_estimates = {'errors': errors, 'quadratic_errors': quadratic_errors}
    obs_stack = jnp.vstack((state/255., next_state/255., neg_state/255.))
    out = jnp.array_split(apply_fn(params_model, obs_stack)[-1], 3)
    (curr_hidden, next_hidden, neg_hidden) = out
    (loss, new_error_estimates), (pos_dot, quadratic_error, neg_dot, neg_norm, pos_loss) =\
                                    proper_laplacian(curr_hidden, next_hidden,
                                    neg_hidden, barrier_coeffs, betas, error_estimates)
    lap_items = [loss, new_error_estimates, pos_dot, neg_dot, neg_norm, pos_loss]

    
    #loss actor critic
    state = state.astype(jnp.float32)/255.

    value_pred, pi, _ = apply_fn(params_model, state)
    value_pred = value_pred[:, 0]

    log_prob = pi.log_prob(action[:,0])

    value_pred_clipped = value_old + (value_pred - value_old).clip(
        -clip_eps, clip_eps)
    value_losses = jnp.square(value_pred - target)
    value_losses_clipped = jnp.square(value_pred_clipped - target)
    value_loss = 0.5 * jnp.maximum(value_losses,
                                   value_losses_clipped).mean()

    ratio = jnp.exp(log_prob - log_pi_old)
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps,
                           1.0 + clip_eps) * gae
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()

    entropy = pi.entropy().mean()

    total_loss = loss_actor + critic_coeff * value_loss - entropy_coeff * entropy

    ppo_items = [total_loss, value_loss, loss_actor, entropy, value_pred.mean(), target.mean(), gae.mean()]
    
    return ppo_items[0] + 1e-6*lap_items[0], (lap_items, ppo_items)



def laplacian_loss_fn(params, apply_fn, obs_i, next_obs_i, neg_obs_i, errors, quadratic_errors, barrier_coeffs, betas): 
        error_estimates = {'errors': errors, 'quadratic_errors': quadratic_errors}
        obs_stack = jnp.vstack((obs_i, next_obs_i, neg_obs_i))
        out = jnp.array_split(apply_fn(params, obs_stack)[-1], 3)
        (curr_hidden, next_hidden, neg_hidden) = out
        (loss, new_error_estimates), (pos_dot, quadratic_error, neg_dot, neg_norm, pos_loss) =\
                                        proper_laplacian(curr_hidden, next_hidden,
                                        neg_hidden, barrier_coeffs, betas, error_estimates)
        return loss, (new_error_estimates, (pos_dot, neg_dot, neg_norm, pos_loss))


def loss_actor_and_critic(
    params_model: flax.core.frozen_dict.FrozenDict,
    apply_fn: Callable[..., Any],
    state: jnp.ndarray,
    target: jnp.ndarray,
    value_old: jnp.ndarray,
    log_pi_old: jnp.ndarray,
    gae: jnp.ndarray,
    action: jnp.ndarray,
    clip_eps: float,
    critic_coeff: float,
    entropy_coeff: float
) -> jnp.ndarray:
    state = state.astype(jnp.float32) / 255.

    value_pred, pi, _ = apply_fn(params_model, state)
    value_pred = value_pred[:, 0]

    log_prob = pi.log_prob(action[:,0])

    value_pred_clipped = value_old + (value_pred - value_old).clip(
        -clip_eps, clip_eps)
    value_losses = jnp.square(value_pred - target)
    value_losses_clipped = jnp.square(value_pred_clipped - target)
    value_loss = 0.5 * jnp.maximum(value_losses,
                                   value_losses_clipped).mean()

    ratio = jnp.exp(log_prob - log_pi_old)
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps,
                           1.0 + clip_eps) * gae
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()

    entropy = pi.entropy().mean()

    total_loss = loss_actor + critic_coeff * value_loss - entropy_coeff * entropy

    return total_loss, (value_loss, loss_actor, entropy, value_pred.mean(),
                        target.mean(), gae.mean())







@partial(jax.jit, static_argnums=0)
def policy(apply_fn: Callable[..., Any],
           params: flax.core.frozen_dict.FrozenDict,
           state: np.ndarray):
    value, pi, _ = apply_fn(params, state)
    return value, pi


def select_action_reg(
    train_state: TrainState,
    state: np.ndarray,
    rng: PRNGKey,
    sample: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, PRNGKey]:
    value, pi = policy(train_state.apply_fn, train_state.params, state)

    if sample:
        rng, key = jax.random.split(rng)
        action = pi.sample(seed=key)
    else:
        action = pi.mode()

    log_prob = pi.log_prob(action)
    return action, log_prob, value[:, 0], rng


def get_transition_reg(
    train_state: TrainState,
    env,
    state,
    batch,
    rng: PRNGKey,
):
    action, log_pi, value, new_key = select_action_reg(
        train_state, state.astype(jnp.float32) / 255., rng, sample=True)
    next_state, reward, done, _ = env.step(action)

    # obs, next_obs, action, reward, done, log_pi, value
    batch.append(state, next_state, action, reward, done, np.array(log_pi), np.array(value))
    return train_state, next_state, batch, new_key



def update_regularizer(
    train_state,
    batch,
    num_envs,
    n_steps,
    n_minibatch,
    epoch_ppo,
    clip_eps,
    entropy_coeff,
    critic_coeff,
    rng
):
    
    # self.obs[:-1], 
    #         self.next_obs[:-1],
    #         self.actions[:-1],
    #         self.log_pis_old[:-1],
    #         self.values_old[:-1],
    #         target,
    #         gae
    state, next_state, action, log_pi_old, value, target, gae = batch

    size_batch = num_envs * n_steps
    size_minibatch = size_batch // n_minibatch

    idxes = np.arange(num_envs * n_steps)

    avg_metrics_dict = defaultdict(int)


    
    for _ in range(epoch_ppo):
        idxes = np.random.permutation(idxes)
        idxes_list = [idxes[start:start + size_minibatch] for start in jnp.arange(0, size_batch, size_minibatch)]
        
        sample_key, rng = jax.jit(jax.random.split)(rng)
        train_state, items = update_jit(
            train_state,
            idxes_list,
            flatten_dims(state),
            flatten_dims(next_state),
            flatten_dims(action),
            flatten_dims(log_pi_old),
            flatten_dims(value),
            np.array(flatten_dims(target)),
            np.array(flatten_dims(gae)),
            clip_eps,
            entropy_coeff,
            critic_coeff,
            sample_key)
        

        lap_ppo_loss = items[0]
        lap_loss, new_error_estimates, pos_dots, neg_dots, neg_norms, pos_losses = items[1][0]

        total_loss, value_loss, loss_actor, entropy, value_pred, target_val, gae_val = items[1][1]

        avg_metrics_dict['total_loss'] += np.asarray(total_loss)
        avg_metrics_dict['value_loss'] += np.asarray(value_loss)
        avg_metrics_dict['actor_loss'] += np.asarray(loss_actor)
        avg_metrics_dict['entropy'] += np.asarray(entropy)
        avg_metrics_dict['value_pred'] += np.asarray(value_pred)
        avg_metrics_dict['target'] += np.asarray(target_val)
        avg_metrics_dict['gae'] += np.asarray(gae_val)
        avg_metrics_dict['pos_dots']=+np.asarray(pos_dots)
        avg_metrics_dict['neg_dots']=+np.asarray(neg_dots)
        avg_metrics_dict['norms']=+np.asarray(neg_norms)
        avg_metrics_dict['pos_loss']=+np.asarray(pos_losses)
        avg_metrics_dict['lap_ppo_loss']=+np.asarray(lap_ppo_loss)
        avg_metrics_dict['lap_loss']=+np.asarray(lap_loss)

    for k, v in avg_metrics_dict.items():
        avg_metrics_dict[k] = v / (epoch_ppo)

    return avg_metrics_dict, train_state, rng


@partial(jax.jit, static_argnames=("clip_eps", "entropy_coeff", "critic_coeff"))
def update_jit(
    train_state: TrainState,
    idxes: np.ndarray,
    state,
    next_state,
    action,
    log_pi_old,
    value,
    target,
    gae,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    key
):
    for idx in idxes:
        # grad_fn = jax.value_and_grad(loss_actor_and_critic,
        #                              has_aux=True)
        # total_loss, grads = grad_fn(train_state.params,
        #                             train_state.apply_fn,
        #                             state=state[idx],
        #                             target=target[idx],
        #                             value_old=value[idx],
        #                             log_pi_old=log_pi_old[idx],
        #                             gae=gae[idx],
        #                             action=action[idx].reshape(-1, 1),
        #                             clip_eps=clip_eps,
        #                             critic_coeff=critic_coeff,
        #                             entropy_coeff=entropy_coeff)
        # train_state = train_state.apply_gradients(grads=grads)


        neg_obs = jax.random.permutation(key, state[idx])
        grad_fn = jax.value_and_grad(combined_loss, has_aux=True)
        items, grads = grad_fn(train_state.params, train_state.apply_fn, state[idx], next_state[idx], neg_obs,
                                   target[idx],
                                    value[idx],
                                    log_pi_old[idx],
                                    gae[idx],
                                    action[idx].reshape(-1, 1),
                                    train_state.errors, train_state.quadratic_errors, train_state.barrier_coeffs, train_state.betas,
                                    clip_eps=clip_eps,
                                    critic_coeff=critic_coeff,
                                    entropy_coeff=entropy_coeff)
        train_state = train_state.apply_gradients(grads=grads)
    return train_state, items

        




