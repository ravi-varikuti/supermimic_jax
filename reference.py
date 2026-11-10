from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

DEFAULT_MOTION = Path(__file__).parent / "LAFAN1_Retargeting_Dataset/g1_npz"


class ReferenceMotion:
    """One NPZ or a directory of clips, concatenated on the JAX device."""

    def __init__(self, path=DEFAULT_MOTION):
        path = Path(path)
        files = sorted(path.glob("*.npz")) if path.is_dir() else [path]
        if not files:
            raise FileNotFoundError(f"No NPZ motions at {path}")
        positions, velocities = [], []
        for file in files:
            with np.load(file, allow_pickle=False) as motion:
                qpos, qvel = motion["qpos"], motion["qvel"]
                fps = float(motion["fps"])
                names = tuple(motion["joint_names"].tolist())
                if str(motion["quaternion_order"]) != "wxyz":
                    raise ValueError("Reference quaternions must use wxyz order")
            if qpos.ndim != 2 or qpos.shape[1] != 36 or qvel.shape != (len(qpos), 35):
                raise ValueError(f"Invalid G1 reference shapes: {file}")
            if len(qpos) < 2 or not np.isfinite(fps) or fps <= 0:
                raise ValueError(f"Reference needs two frames and positive FPS: {file}")
            if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
                raise ValueError(f"Nonfinite reference: {file}")
            if positions and (fps != self.fps or names != self.joint_names):
                raise ValueError("All clips must have matching FPS and joint order")
            self.fps, self.joint_names = fps, names
            positions.append(qpos)
            velocities.append(qvel)
        self.motion_names = tuple(file.stem for file in files)
        self.num_motions = len(files)
        self._lengths = np.array([len(q) for q in positions], dtype=np.int32)
        self.lengths = jnp.asarray(self._lengths)
        self.offsets = jnp.asarray(np.r_[0, np.cumsum(self._lengths)[:-1]], dtype=jnp.int32)
        self.qpos = jnp.asarray(np.concatenate(positions), dtype=jnp.float32)
        self.qvel = jnp.asarray(np.concatenate(velocities), dtype=jnp.float32)
        self.num_frames = len(self.qpos)  # Total across all clips.

    def get(self, frames, motion_ids=0):
        """Look up clip-local frames, clamping at each clip's boundary."""
        frames = jnp.asarray(frames, dtype=jnp.int32)
        motion_ids = jnp.asarray(motion_ids, dtype=jnp.int32)
        frames = jnp.clip(frames, 0, self.lengths[motion_ids] - 1)
        indices = self.offsets[motion_ids] + frames
        return self.qpos[indices], self.qvel[indices]

    def sample_jax(self, key, count, motion_ids=None, frames=None):
        """Fixed-shape device sampling; explicit overrides must already be valid."""
        motion_key, frame_key = jax.random.split(key)
        ids = jax.random.randint(motion_key, (count,), 0, self.num_motions)
        if motion_ids is not None:
            ids = jnp.broadcast_to(jnp.asarray(motion_ids, jnp.int32), (count,))
        starts = jax.random.randint(frame_key, (count,), 0, self.lengths[ids] - 1)
        if frames is not None:
            starts = jnp.broadcast_to(jnp.asarray(frames, jnp.int32), (count,))
        return ids, starts
