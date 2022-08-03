"""Create a randomly initialized policy network.
"""
import sys
from os import environ
BASE_PATH = environ.get("DRL_BASE", "")
sys.path.insert(0, BASE_PATH)
import torch as pt
from drlfoam.agent import FCPolicy


policy = FCPolicy(12, 1, -5.0, 5.0)
script = pt.jit.script(policy)
script.save("random_policy.pt")