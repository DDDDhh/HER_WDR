# -*- coding: utf-8 -*-
"""
Hindsight Experience Replay (HER) for parafoil upwind flare landing.

Design principles
-----------------
1. **Pure functions.** All HER computations operate on numpy arrays of stored
   physical states. They do NOT mutate the env, the env's args, the storage,
   or any shared memory outside the explicit return value. This avoids the
   fragile proxy / args-mutation patterns of earlier attempts.

2. **Goal-as-wind-direction.** For upwind flare landing the implicit goal is
   the mean wind direction (because target heading = wind_dir + pi). The HER
   relabel strategy is "final": the achieved final heading is treated as the
   new "perfect upwind" heading, i.e. hindsight wind direction is set so that
   the final heading equals wind_dir + pi.

3. **Consistent units.** All angles inside HER are RADIANS. wind_dir extracted
   from the stored wind vector ``status[28:30]`` via atan2 is naturally rad,
   so the agent never sees a degree/radian mismatch.

The HER relabel rotates wind direction by ``delta = hindsight_wind_dir -
real_wind_dir``. Since parafoil body-frame dynamics are invariant under wind
rotation (wind only shifts the ground-frame trajectory by a rotation), the
relabeled trajectory remains physically self-consistent for body-frame
observations and reward components.
"""
import numpy as np

from parafoil_env.utils.method import angle_normalize


def extract_wind_dir_rad(status):
    """Return wind direction (rad) from a physical status row.

    Uses the stored ground-frame wind vector ``status[28:30]`` (m/s). Falls
    back to 0.0 if the wind is essentially zero.
    """
    wx, wy = float(status[28]), float(status[29])
    if abs(wx) < 1e-8 and abs(wy) < 1e-8:
        return 0.0
    return float(np.arctan2(wy, wx))


def rotate_wind_vector(wind_xyz, delta_rad):
    """Rotate a 3D wind vector around the z-axis by ``delta_rad`` (rad)."""
    c, s = np.cos(delta_rad), np.sin(delta_rad)
    wx, wy, wz = float(wind_xyz[0]), float(wind_xyz[1]), float(wind_xyz[2])
    return np.array([c * wx - s * wy, s * wx + c * wy, wz], dtype=np.float32)


def compute_hindsight_wind_dir(final_heading_rad):
    """Hindsight strategy: choose wind_dir so the final heading is perfectly upwind.

    target_heading = wind_dir + pi  =>  wind_dir = final_heading - pi
    """
    return float(angle_normalize(final_heading_rad - np.pi, unit='rad'))


def relabel_phys_states_inplace(phys_states, real_wind_dir_rad, hindsight_wind_dir_rad):
    """Rotate the wind vector in each stored phys state to the hindsight wind dir.

    Operates on a COPY of the original phys states. Returns the new list.
    """
    delta = float(angle_normalize(hindsight_wind_dir_rad - real_wind_dir_rad, unit='rad'))
    out = []
    for s in phys_states:
        s_new = np.copy(s)
        s_new[28:31] = rotate_wind_vector(s_new[28:31], delta)
        out.append(s_new)
    return out


def relabel_trajectory(env_pure, phys_states, action_rate, max_episode_steps,
                       reward_kwargs=None, noise_std=0.0, rs=None):
    """Given a list of (rate+1) phys states per RL step, compute relabeled
    (observations, rewards) under the hindsight wind direction.

    Parameters
    ----------
    env_pure : object exposing the static helpers
        ``compute_observation(status, wind_dir_rad)`` and
        ``compute_reward(status_now, status_prev, wind_dir_rad, elapsed_steps,
                         max_episode_steps, reward_kwargs)``
        — EnvFlare provides these as @staticmethod.
    phys_states : list[np.ndarray]
        len = K+1, where K is the number of RL steps taken in the episode.
        ``phys_states[0]`` is the initial state, ``phys_states[k]`` is the
        phys state at the end of RL step k.
    action_rate : int
        Number of sim steps per RL step (informational only; not used for
        indexing here because phys_states is already step-indexed).
    max_episode_steps : int
    reward_kwargs : dict, optional
    noise_std : float, optional
        Standard deviation of Gaussian noise (radians) added to the hindsight
        wind direction. Default 0.0 (exact relabeling = standard WDR).
        Set > 0 for WDR noise ablation experiments.
    rs : numpy.random.RandomState, optional
        Random state for reproducible noise injection.

    Returns
    -------
    dict with keys:
      - 'obs' : np.ndarray (K+1, obs_dim)
      - 'rewards' : np.ndarray (K,)
      - 'hindsight_wind_dir_rad' : float
      - 'real_wind_dir_rad' : float
      - 'final_done_type' : str ('complete' or 'dead'/'done' if rule says so)
    """
    assert len(phys_states) >= 2, 'need at least initial + one step for HER'
    K = len(phys_states) - 1

    final_heading = float(phys_states[-1][8])
    real_wind_dir = extract_wind_dir_rad(phys_states[-1])
    hindsight_wind_dir = compute_hindsight_wind_dir(final_heading)

    # WDR noise injection: add Gaussian noise to the hindsight wind direction
    if noise_std > 0.0:
        rng = rs if rs is not None else np.random
        noise = float(rng.normal(0.0, noise_std))
        hindsight_wind_dir = float(angle_normalize(hindsight_wind_dir + noise, unit='rad'))

    # Relabel stored phys states (rotated wind vector). Important so that the
    # body-frame wind component in the observation matches the new wind dir.
    relabeled = relabel_phys_states_inplace(phys_states, real_wind_dir, hindsight_wind_dir)

    # New observations at every state boundary (K+1 of them)
    obs_list = [env_pure.compute_observation(s, hindsight_wind_dir) for s in relabeled]
    obs_arr = np.stack(obs_list, axis=0).astype(np.float32)

    # New rewards for each of K RL transitions (state[k] -> state[k+1])
    rewards = np.zeros(K, dtype=np.float32)
    reward_kwargs = reward_kwargs or {}
    for k in range(K):
        prev = relabeled[k]
        now = relabeled[k + 1]
        r, _done, _info = env_pure.compute_reward(
            status_now=now,
            status_prev=prev,
            wind_dir_rad=hindsight_wind_dir,
            elapsed_steps=k + 1,
            max_episode_steps=max_episode_steps,
            reward_kwargs=reward_kwargs,
        )
        rewards[k] = r

    return {
        'obs': obs_arr,
        'rewards': rewards,
        'hindsight_wind_dir_rad': hindsight_wind_dir,
        'real_wind_dir_rad': real_wind_dir,
    }


def should_apply_her(phys_states, done_type, min_steps=5, min_alt_descended=20.0,
                     trigger_prob=1.0, rs=None):
    """Decide whether to apply HER on a finished episode.

    Heuristics:
    - Skip if too short (less than ``min_steps`` RL steps).
    - Skip on real successes (``done_type == 'complete'``) — already labelled.
    - Skip if the parafoil barely descended (probably crashed at init).
    - Otherwise apply with probability ``trigger_prob``.
    """
    if done_type == 'complete':
        return False
    K = len(phys_states) - 1
    if K < int(min_steps):
        return False
    init_alt = -float(phys_states[0][2])
    final_alt = -float(phys_states[-1][2])
    if (init_alt - final_alt) < float(min_alt_descended):
        return False
    rng = rs if rs is not None else np.random
    return bool(rng.random() < float(trigger_prob))
