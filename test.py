import time
from rl_env import G1Env


if __name__ == "__main__":
    env = G1Env(num_envs=16)
    dt = env.sim.model.opt.timestep * env.sim.decimation
    joint_ids = env.sim.model.actuator_trnid[:, 0]
    targets = env.sim.data.qpos[:, env.sim.model.jnt_qposadr[joint_ids]]

    try:
        env.render()
        input("Open http://127.0.0.1:8080, then press Enter to start.")

        for _ in range(round(10.0 / dt)):
            start = time.perf_counter()
            env.step(targets)  # Hold the initial joint targets.
            env.render()
            time.sleep(max(0.0, dt - (time.perf_counter() - start)))

        input("Finished 10 seconds of simulation. Press Enter to close.")
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
