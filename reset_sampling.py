"""Return-based motion resets with device sampling and host episode accounting."""

import jax
import jax.numpy as jnp
import numpy as np


class ReturnSampler:
    def __init__(self, lengths, fps, num_envs, max_episode_steps, discount=.99,
                 max_reward=.1, smoothing=.1):
        if not 0 <= discount < 1 or not 0 < smoothing <= 1:
            raise ValueError("Invalid return-sampler discount or smoothing")
        self.discount, self.smoothing = discount, smoothing
        self.max_return = max_reward / (1. - discount)
        self.fps = float(fps)
        counts = np.ceil((np.asarray(lengths) - 1) / fps).astype(int)
        self.offsets = np.r_[0, np.cumsum(counts)[:-1]]
        ids = np.repeat(np.arange(len(counts)), counts)
        seconds = np.concatenate([np.arange(count) for count in counts])
        low = np.ceil(seconds * fps).astype(np.int32)
        high = np.minimum(np.ceil((seconds + 1) * fps).astype(np.int32), np.asarray(lengths)[ids] - 1)
        self.ids, self.low, self.high = map(jnp.asarray, (ids.astype(np.int32), low, high))
        size = len(ids)
        self.state = {
            "estimate": np.zeros(size, np.float32), "visits": np.zeros(size, np.int64),
            "past_weight": np.ones(size, np.float64),
            "pending_rewards": np.zeros((num_envs, max_episode_steps), np.float32),
            "pending_bins": np.zeros((num_envs, max_episode_steps), np.int32),
            "pending_length": np.zeros(num_envs, np.int32),
        }

    def probabilities(self):
        score = -25. * self.state["estimate"] / self.max_return
        hard = np.exp(score - score.max())
        hard /= hard.sum()
        past = self.state["past_weight"] / self.state["past_weight"].sum()
        return jnp.asarray(.5 / len(hard) + .4 * hard + .1 * past, jnp.float32)

    def sample_jax(self, key, probabilities, count):
        bin_key, frame_key = jax.random.split(key)
        # Inverse CDF avoids count x number-of-bins categorical allocations.
        bins = jnp.searchsorted(jnp.cumsum(probabilities), jax.random.uniform(bin_key, (count,)), side="right")
        bins = jnp.minimum(bins, len(self.ids) - 1)
        frames = jax.random.randint(frame_key, (count,), self.low[bins], self.high[bins])
        return self.ids[bins], frames

    def observe(self, records):
        """Complete discounted returns across collection chunks; update only at episode end.

        EMA speed and accumulation of historical weights are implementation choices:
        the paper specifies the mixture, but not those estimator details.
        """
        data = jax.device_get(records)
        done = np.asarray(data.terminated) | np.asarray(data.truncated)
        state = self.state
        sums = np.zeros_like(state["estimate"], dtype=np.float64)
        counts = np.zeros_like(state["visits"])
        for t in range(len(data.reward)):
            for world in range(data.reward.shape[1]):
                step = int(data.metadata["episode_steps"][t, world])
                if step == 0:
                    state["pending_length"][world] = 0
                if step != state["pending_length"][world]:
                    raise ValueError("Return sampling requires uninterrupted episode records")
                if step >= state["pending_rewards"].shape[1]:
                    raise ValueError("Episode exceeds return-sampler capacity")
                frame = int(data.metadata["reference_frame"][t, world])
                motion = int(data.metadata["motion_id"][t, world])
                bin_id = self.offsets[motion] + int(frame / self.fps)
                state["pending_rewards"][world, step] = np.nan_to_num(
                    data.reward[t, world], nan=0., posinf=0., neginf=0.)
                state["pending_bins"][world, step] = bin_id
                state["pending_length"][world] = step + 1
                if done[t, world]:
                    value = 0.
                    for index in range(step, -1, -1):
                        value = float(state["pending_rewards"][world, index]) + self.discount * value
                        slot = state["pending_bins"][world, index]
                        sums[slot] += value
                        counts[slot] += 1
                    state["pending_length"][world] = 0
        observed = counts > 0
        mean = sums / np.maximum(counts, 1)
        fresh = state["visits"] == 0
        speed = np.where(fresh, 1., self.smoothing)
        state["estimate"][observed] += (speed * (mean - state["estimate"]))[observed]
        state["visits"] += counts
        score = -25. * state["estimate"] / self.max_return
        hard = np.exp(score - score.max())
        state["past_weight"] += hard / hard.sum()
