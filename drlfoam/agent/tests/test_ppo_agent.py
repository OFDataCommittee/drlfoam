

from os import remove
from os.path import join, isfile
import torch as pt
from ..ppo_agent import PPOAgent
from ...constants import DEFAULT_TENSOR_TYPE

pt.set_default_tensor_type(DEFAULT_TENSOR_TYPE)


class TestPPOAgent():
    def test_update(self):
        states = [pt.rand((10, 100)) for _ in range(5)]
        rewards = [pt.rand(10) for _ in range(5)]
        actions = [pt.rand(10) for _ in range(5)]
        agent = PPOAgent(100, 1, pt.tensor(-10), pt.tensor(10))
        agent.update(states, actions, rewards)
        hist = agent.history
        assert "policy_loss" in hist
        assert len(hist["policy_loss"]) == 1
        assert "value_loss" in hist
        assert len(hist["value_loss"]) == 1

    def test_save_load(self):
        states = pt.rand((5, 100))
        agent = PPOAgent(100, 1, pt.tensor(-10), pt.tensor(10))
        p_out_ref = agent._policy(states)
        v_out_ref = agent._value(states)
        checkpoint_path = join("/tmp", "checkpoint.pt")
        agent.save_state(checkpoint_path)
        assert isfile(checkpoint_path)
        agent.load_state(checkpoint_path)
        p_out = agent._policy(states)
        v_out = agent._value(states)
        assert pt.allclose(p_out_ref, p_out)
        assert pt.allclose(v_out_ref, v_out)

    def test_trace_policy(self):
        states = pt.rand((5, 100))
        agent = PPOAgent(100, 1, pt.tensor(-10), pt.tensor(10))
        p_out_ref = agent._policy(states)
        trace = agent.trace_policy()
        p_out = trace(states)
        assert pt.allclose(p_out_ref, p_out)
