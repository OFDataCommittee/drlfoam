""" Example training script.
"""

import argparse
from shutil import copytree
from os.path import join
from os import makedirs
import sys
from os import environ
from time import time
import logging
BASE_PATH = environ.get("DRL_BASE", "")
sys.path.insert(0, BASE_PATH)

import torch as pt
from drlfoam.environment import RotatingCylinder2D, RotatingPinball2D
from drlfoam.agent import PPOAgent
from drlfoam.execution import LocalBuffer, SlurmBuffer, SlurmConfig


logging.basicConfig(level=logging.INFO)


SIMULATION_ENVIRONMENTS = {
    "rotatingCylinder2D" : RotatingCylinder2D,
    "rotatingPinball2D" : RotatingPinball2D
}

DEFAULT_CONFIG = {
    "rotatingCylinder2D" : {
        "policy_dict" : {
            "n_layers": 2,
            "n_neurons": 64,
            "activation": pt.nn.functional.relu
        },
        "value_dict" : {
            "n_layers": 2,
            "n_neurons": 64,
            "activation": pt.nn.functional.relu
        }
    },
    "rotatingPinball2D" : {
        "policy_dict" : {
            "n_layers": 2,
            "n_neurons": 512,
            "activation": pt.nn.functional.relu
        },
        "value_dict" : {
            "n_layers": 2,
            "n_neurons": 512,
            "activation": pt.nn.functional.relu
        },
        "policy_lr" : 4.0e-4,
        "value_lr" : 4.0e-4
    }
}


def print_statistics(actions, rewards):
    rt = [r.mean().item() for r in rewards]
    at_mean = [a.mean().item() for a in actions]
    at_std = [a.std().item() for a in actions]
    reward_msg = f"Reward mean/min/max: {sum(rt)/len(rt):2.4f}/{min(rt):2.4f}/{max(rt):2.4f}"
    action_mean_msg = f"Mean action mean/min/max: {sum(at_mean)/len(at_mean):2.4f}/{min(at_mean):2.4f}/{max(at_mean):2.4f}"
    action_std_msg = f"Std. action mean/min/max: {sum(at_std)/len(at_std):2.4f}/{min(at_std):2.4f}/{max(at_std):2.4f}"
    logging.info("\n".join((reward_msg, action_mean_msg, action_std_msg)))


def parseArguments():
    ag = argparse.ArgumentParser()
    ag.add_argument("-o", "--output", required=False, default="test_training", type=str,
                    help="Where to run the training.")
    ag.add_argument("-e", "--environment", required=False, default="local", type=str,
                    help="Use 'local' for local and 'slurm' for cluster execution.")
    ag.add_argument("-i", "--iter", required=False, default=20, type=int,
                    help="Number of training episodes.")
    ag.add_argument("-r", "--runners", required=False, default=4, type=int,
                    help="Number of runners for parallel execution.")
    ag.add_argument("-b", "--buffer", required=False, default=8, type=int,
                    help="Reply buffer size.")
    ag.add_argument("-f", "--finish", required=False, default=8.0, type=float,
                    help="End time of the simulations.")
    ag.add_argument("-t", "--timeout", required=False, default=1e15, type=int,
                    help="Maximum allowed runtime of a single simulation in seconds.")
    ag.add_argument("-c", "--checkpoint", required=False, default="", type=str,
                    help="Load training state from checkpoint file.")
    ag.add_argument("-s", "--simulation", required=False, default="rotatingCylinder2D", type=str,
                    help="Select the simulation environment.")
    args = ag.parse_args()
    return args


def main(args):
    # settings
    training_path = args.output
    episodes = args.iter
    buffer_size = args.buffer
    n_runners = args.runners
    end_time = args.finish
    executer = args.environment
    timeout = args.timeout
    checkpoint_file = args.checkpoint
    simulation = args.simulation

    # create a directory for training
    makedirs(training_path, exist_ok=True)

    # make a copy of the base environment
    if not simulation in SIMULATION_ENVIRONMENTS.keys():
        msg = (f"Unknown simulation environment {simulation}" +
              "Available options are:\n\n" +
              "\n".join(SIMULATION_ENVIRONMENTS.keys()) + "\n")
        raise ValueError(msg)
    copytree(join(BASE_PATH, "openfoam", "test_cases", simulation),
             join(training_path, "base"), dirs_exist_ok=True)
    env = SIMULATION_ENVIRONMENTS[simulation]()
    env.path = join(training_path, "base")

    # create buffer
    if executer == "local":
        buffer = LocalBuffer(training_path, env, buffer_size, n_runners, timeout=timeout)
    elif executer == "slurm":
        # Typical Slurm configs for TU Braunschweig cluster
        config = SlurmConfig(
            n_tasks=env.mpi_ranks, n_nodes=1, partition="queue-1", time="03:00:00",
            constraint="c5a.24xlarge", modules=["openmpi/4.1.5"],
            commands_pre=["source /fsx/OpenFOAM/OpenFOAM-v2206/etc/bashrc", "source /fsx/drlfoam_main/setup-env"]
        )
        buffer = SlurmBuffer(training_path, env,
                             buffer_size, n_runners, config, timeout=timeout)
    else:
        raise ValueError(
            f"Unknown executer {executer}; available options are 'local' and 'slurm'.")

    # create PPO agent
    agent = PPOAgent(env.n_states, env.n_actions, -env.action_bounds, env.action_bounds,
                     **DEFAULT_CONFIG[simulation])

    # load checkpoint if provided
    if checkpoint_file:
        logging.info(f"Loading checkpoint from file {checkpoint_file}")
        agent.load_state(join(training_path, checkpoint_file))
        starting_episode = agent.history["episode"][-1] + 1
        buffer._n_fills = starting_episode
    else:
        starting_episode = 0
        buffer.prepare()

    buffer.base_env.start_time = buffer.base_env.end_time
    buffer.base_env.end_time = end_time
    buffer.reset()

    # begin training
    start_time = time()
    for e in range(starting_episode, episodes):
        logging.info(f"Start of episode {e}")
        buffer.fill()
        states, actions, rewards = buffer.observations
        print_statistics(actions, rewards)
        agent.update(states, actions, rewards)
        agent.save_state(join(training_path, f"checkpoint_{e}.pt"))
        current_policy = agent.trace_policy()
        buffer.update_policy(current_policy)
        current_policy.save(join(training_path, f"policy_trace_{e}.pt"))
        if not e == episodes - 1:
            buffer.reset()
    logging.info(f"Training time (s): {time() - start_time}")


if __name__ == "__main__":
    main(parseArguments())
