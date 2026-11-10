"""Small end-to-end check of corrected control, features, normalization and updates."""
import jax
import jax.numpy as jnp
from train import TrainingRun
from train_config import preset
from reference import DEFAULT_MOTION

if __name__ == '__main__':
    run = TrainingRun(preset('smoke'), DEFAULT_MOTION)
    run.warmup()
    sample = run.sample_actor(run.state.replay, jax.random.key(8))
    assert bool(sample.valid)
    records = sample.transitions
    reference = run.env.reference_from_metadata(records.metadata, next_frame=True)
    observed = jax.jit(run.env.observation_model.clean)(records.next_state, records.action,
                                                      records.context, reference)
    assert bool(jnp.allclose(observed, records.next_observation, atol=1e-5))
    reward, _ = jax.jit(run.env.reward_model)(records.next_state, reference, records.action, records.previous_action)
    assert bool(jnp.allclose(reward, records.reward, atol=1e-5))
    old_actor = run.state.actor_critic.actor.params
    metrics = run.iteration()
    assert metrics['updated'] and metrics['world_model_updated']
    assert float(run.state.actor_critic.normalization.observation.count) > 0
    assert float(run.state.dynamics.normalization.count) > 0
    assert any(bool(jnp.any(a != b)) for a,b in zip(jax.tree.leaves(old_actor),
                                                   jax.tree.leaves(run.state.actor_critic.actor.params)))
    print('PASS: 99D observations and pelvis rewards match Warp records')
    print('PASS: running normalization and local-model actor/critic update', metrics)
