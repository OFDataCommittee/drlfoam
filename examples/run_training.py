""" Example training script.
"""



import argparse
from shutil import copytree
import pickle
from os.path import join
from os import makedirs
import sys
from os import environ
from time import time
BASE_PATH = environ.get("DRL_BASE", "")
sys.path.insert(0, BASE_PATH)

from drlfoam.environment import RotatingCylinder2D
from drlfoam.agent import PPOAgent
from drlfoam.execution import LocalBuffer, SlurmBuffer, SlurmConfig


def print_statistics(actions, rewards):
    rt = [r.mean().item() for r in rewards]
    at_mean = [a.mean().item() for a in actions]
    at_std = [a.std().item() for a in actions]
    print("Reward mean/min/max: ", sum(rt)/len(rt), min(rt), max(rt))
    print("Mean action mean/min/max: ", sum(at_mean) /
          len(at_mean), min(at_mean), max(at_mean))
    print("Std. action mean/min/max: ", sum(at_std) /
          len(at_std), min(at_std), max(at_std))


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

    # create a directory for training
    makedirs(training_path, exist_ok=True)

    # make a copy of the base environment
    copytree(join(BASE_PATH, "openfoam", "test_cases", "rotatingCylinder2D"),
             join(training_path, "base"), dirs_exist_ok=True)
    env = RotatingCylinder2D()
    env.path = join(training_path, "base")

    # create buffer
    if executer == "local":
        buffer = LocalBuffer(training_path, env, buffer_size, n_runners, timeout=timeout)
    elif executer == "slurm":
        # Typical Slurm configs for TU Braunschweig cluster
        config = SlurmConfig(
            n_tasks=2, n_nodes=1, partition="standard", time="00:30:00",
            modules=["singularity/latest", "mpi/openmpi/4.1.1/gcc"]
        )
        buffer = SlurmBuffer(training_path, env,
                             buffer_size, n_runners, config, timeout=timeout)
    else:
        raise ValueError(
            f"Unknown executer {executer}; available options are 'local' and 'slurm'.")

    # execute Allrun.pre script and set new end_time
    buffer.prepare()
    buffer.base_env.start_time = buffer.base_env.end_time
    buffer.base_env.end_time = end_time
    buffer.reset()

    # create PPO agent
    agent = PPOAgent(env.n_states, env.n_actions, -
                     env.action_bounds, env.action_bounds)

    # begin training
    start_time = time()
    for e in range(episodes):
        print(f"Start of episode {e}")
        buffer.fill()
        states, actions, rewards = buffer.observations
        print_statistics(actions, rewards)
        agent.update(states, actions, rewards)
        agent.save(join(training_path, f"policy_{e}.pkl"),
                   join(training_path, f"value_{e}.pkl"))
        current_policy = agent.trace_policy()
        buffer.update_policy(current_policy)
        current_policy.save(join(training_path, f"policy_trace_{e}.pt"))
        buffer.reset()
    print(f"Training time (s): {time() - start_time}")

    # save training statistics
    with open(join(training_path, "training_history.pkl"), "wb") as f:
        pickle.dump(agent.history, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main(parseArguments())
