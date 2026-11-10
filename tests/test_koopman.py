import unittest
from types import SimpleNamespace
from flax.core import unfreeze
import jax
import jax.numpy as jnp
import numpy as np

from configs.g1 import G1_CONFIG
from geometry import exp_rotation, invariant_state, rotation6d
from koopman import Koopman, KoopmanConfig
from koopman_g1 import KoopmanWorldModel, quaternion_from_6d, reconstruct
from normalization import FeatureStats, Normalization
from world_model_training import WorldModelTrainer


def stats():
    return Normalization(FeatureStats(jnp.zeros(71),jnp.ones(71)),
                         FeatureStats(jnp.zeros(33),jnp.ones(33)),
                         FeatureStats(jnp.zeros(70),jnp.ones(70)),jnp.array(1.))


def states(batch=2):
    return jnp.zeros((batch,71)).at[:,2].set(.8).at[:,3].set(1.)


class KoopmanTests(unittest.TestCase):
    def test_configurable_dimensions_fixed_projection_and_initialization(self):
        c=KoopmanConfig(state_dim=5,action_dim=2,context_dim=3,feature_dim=7,encoder_hidden=(8,))
        model=Koopman(c)
        x,a,ctx=jnp.ones((4,5)),jnp.ones((4,2)),jnp.ones((4,3))
        v=model.init(jax.random.key(1),x,a,ctx)
        self.assertEqual(set(v['params']), {'encoder','A','B','C'})
        z=model.apply(v,x,method=model.encode)
        np.testing.assert_array_equal(z[:,:5],x)
        np.testing.assert_array_equal(model.apply(v,z,method=model.decode),x)
        self.assertLess(float(jnp.max(jnp.abs(v['params']['A']-jnp.eye(12)))),.01)
        self.assertGreater(float(jnp.linalg.norm(v['params']['B'])),0.)
        self.assertEqual(model.apply(v,x,a,ctx).shape,(4,5))
        plain=Koopman(KoopmanConfig(state_dim=2,action_dim=1,context_dim=0,feature_dim=2,encoder_hidden=()))
        self.assertEqual(plain.init(jax.random.key(3),jnp.zeros((1,2)),jnp.zeros((1,1)),jnp.zeros((1,0)))['params']['C'].shape,(4,0))

    def test_scan_carries_latent_and_context_without_reencoding(self):
        c=KoopmanConfig(state_dim=2,action_dim=1,context_dim=1,feature_dim=1,encoder_hidden=())
        model=Koopman(c)
        x=jnp.zeros((1,2)); a=jnp.ones((1,8,1)); ctx=2*jnp.ones((1,8,1))
        v=unfreeze(model.init(jax.random.key(1),x,a[:,0],ctx[:,0]))
        p=v['params']; p['encoder']['output']['kernel']=jnp.zeros((2,1)); p['encoder']['output']['bias']=jnp.zeros(1)
        p['A']=jnp.eye(3).at[0,2].set(1.)
        p['B']=jnp.array([[0.],[0.],[1.]])
        p['C']=jnp.array([[0.],[.5],[0.]])
        projected,z=jax.jit(lambda x,a,c:model.apply(v,x,a,c,method=model.rollout))(x,a,ctx)
        n=np.arange(1,9)
        np.testing.assert_allclose(projected[0,:,0],n*(n-1)/2)
        np.testing.assert_allclose(projected[0,:,1],n)
        np.testing.assert_allclose(z[0,:,2],n)
        grad=jax.grad(lambda acts:model.apply(v,x,acts,ctx,method=model.rollout)[0].sum())(a)
        self.assertGreater(float(jnp.linalg.norm(grad)),0.)

    def test_rotation_projection_reconstruction_and_gradients(self):
        q=exp_rotation(jnp.array([[.1,.2,.3],[0.,0.,3.14]]))
        recovered=quaternion_from_6d(rotation6d(q))
        np.testing.assert_allclose(jnp.abs(jnp.sum(q*recovered,-1)),1.,atol=1e-6)
        for six in (jnp.zeros(6),jnp.array([1.,0.,0.,0.,1.,0.])):
            self.assertTrue(bool(jnp.isfinite(jax.jacrev(quaternion_from_6d)(six)).all()))
        s=states(1).at[:,36].set(1.).at[:,41].set(.5)
        features=invariant_state(s)
        after=reconstruct(s,features,.02)
        np.testing.assert_allclose(after[:,0],.02,atol=1e-5)
        np.testing.assert_allclose(after[:,2],.8)
        np.testing.assert_allclose(jnp.linalg.norm(after[:,3:7],axis=-1),1.,atol=1e-6)
        self.assertGreater(float(after[0,6]),0.)
        self.assertTrue(bool(jnp.isfinite(jax.jacrev(lambda f:reconstruct(s,f,.02))(features)).all()))

    def test_g1_losses_and_updates_use_distinct_one_step_and_open_loop_targets(self):
        c=KoopmanConfig(feature_dim=4,encoder_hidden=(8,),latent_weight=0.)
        model=KoopmanWorldModel(G1_CONFIG,c)
        trainer=WorldModelTrainer(model,learning_rate=.001)
        training=trainer.init(jax.random.key(2),stats())
        p=unfreeze(training.params)
        p['core']['A']=jnp.eye(75); p['core']['B']=jnp.zeros((75,29)); p['core']['C']=jnp.zeros((75,33))
        s=jnp.repeat(states()[:,None],8,axis=1)
        following=s.at[:,:,7].set(jnp.arange(1,9)*.05)
        current=s.at[:,1:,7].set(jnp.arange(1,8)*.05)
        records=SimpleNamespace(state=current,next_state=following,action=jnp.zeros((2,8,29)),context=jnp.zeros((2,8,33)))
        loss,terms=trainer.koopman_loss(p,stats(),records)
        self.assertGreater(float(terms['multi_step_loss']),float(terms['one_step_loss'])*10)
        self.assertEqual(float(terms['latent_loss']),0.)
        np.testing.assert_allclose(loss,terms['one_step_loss']+terms['multi_step_loss'])
        before=float(loss)
        def fit(params):
            grads=jax.grad(lambda params:trainer.koopman_loss(params,stats(),records)[0])(params)
            return jax.tree.map(lambda p,g:p-.01*g,params,grads)
        update=jax.jit(fit)
        for _ in range(30): p=update(p)
        self.assertLess(float(trainer.koopman_loss(p,stats(),records)[0]),before*.5)
        # Later true input states influence only the teacher-forced term.
        altered=SimpleNamespace(**(vars(records)|{'state':current.at[:,1:,7].add(.5)}))
        changed=trainer.koopman_loss(p,stats(),altered)[1]
        original=trainer.koopman_loss(p,stats(),records)[1]
        np.testing.assert_allclose(changed['multi_step_loss'],original['multi_step_loss'])
        self.assertNotAlmostEqual(float(changed['one_step_loss']),float(original['one_step_loss']))
        active=KoopmanWorldModel(G1_CONFIG,KoopmanConfig(feature_dim=4,encoder_hidden=(8,),latent_weight=.1))
        active_trainer=WorldModelTrainer(active)
        _,details=active_trainer.koopman_loss(p,stats(),records)
        self.assertGreater(float(details['latent_loss']),0.)
        physical,latent=jax.jit(lambda:active.apply({'params':p},current[:,0],records.action,records.context,stats(),method=active.rollout))()
        self.assertEqual(physical.shape,(2,8,71)); self.assertEqual(latent.shape,(2,8,75))
        np.testing.assert_allclose(jnp.linalg.norm(physical[...,3:7],axis=-1),1.,atol=1e-6)


if __name__=='__main__': unittest.main()
