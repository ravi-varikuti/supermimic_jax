import time
import numpy as np
from rl_env import G1Env


if __name__ == "__main__":
    env = G1Env(num_envs=16)
    observation, info = env.reset(seed=0)
    print("Reset observation shape:", observation.shape)
    dt = env.sim.model.opt.timestep * env.sim.decimation
    action = env.previous_action  # Zero action commands the nominal joint pose.

    try:
        env.render()
        input("Open http://127.0.0.1:8080, then press Enter to start.")

        for _ in range(round(10.0 / dt)):
            start = time.perf_counter()
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            done_ids = np.flatnonzero(np.asarray(terminated | truncated))
            if done_ids.size:
                observation, info = env.reset(env_ids=done_ids)
            time.sleep(max(0.0, dt - (time.perf_counter() - start)))

        print("Final mean reward:", float(reward.mean()))
        input("Finished 10 seconds of simulation. Press Enter to close.")
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
