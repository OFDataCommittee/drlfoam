
from os import makedirs, remove
from os.path import join, isdir, isfile
from shutil import copytree, rmtree
from copy import deepcopy
from pytest import fixture, raises
import torch as pt
from ..local import LocalBuffer
from ...environment import RotatingCylinder2D
from ...agent import FCPolicy
from ...constants import TESTDATA_PATH, TESTCASE_PATH


@fixture()
def temp_training():
    training = join("/tmp", "test_training")
    makedirs(training, exist_ok=True)
    case = "rotatingCylinder2D"
    source = join(TESTCASE_PATH, case)
    dest = join(training, case)
    copytree(source, dest, dirs_exist_ok=True)
    env = RotatingCylinder2D()
    env.path = dest
    yield (training, env)
    rmtree(training)


class TestLocalBuffer():
    def test_create_copies(self, temp_training):
        path, base_env = temp_training
        buffer = LocalBuffer(path, base_env, 2, 2)
        envs = buffer.envs
        assert isdir(join(path, "copy_0"))
        assert isdir(join(path, "copy_1"))
        assert envs[0].path == join(path, "copy_0")
        assert envs[1].path == join(path, "copy_1")

    def test_update_policy(self, temp_training):
        path, base_env = temp_training
        buffer = LocalBuffer(path, base_env, 1, 1)
        envs = buffer.envs
        remove(join(envs[0].path, envs[0].policy))
        policy = FCPolicy(1, 1, -1, 1)
        buffer.update_policy(pt.jit.script(policy))
        assert isfile(join(envs[0].path, envs[0].policy))

    def test_prepare_fill_reset_clean(self, temp_training):
        path, base_env = temp_training
        base_env.start_time = 0.0
        base_env.end_time = 0.015
        buffer = LocalBuffer(path, base_env, 2, 2)
        buffer.prepare()
        assert isfile(join(path, base_env.path, "log.blockMesh"))
        assert isfile(join(path, base_env.path, "trajectory.csv"))
        assert isdir(join(path, base_env.path, "postProcessing"))
        buffer.reset()
        assert not isfile(join(path, "copy_0", "trajectory.csv"))
        assert not isdir(join(path, "copy_0", "postProcessing"))
        buffer.fill()
        assert isfile(join(path, "copy_0", "trajectory.csv"))
        assert isfile(join(path, "copy_0", "log.pimpleFoam"))
        assert isdir(join(path, "copy_0", "postProcessing"))
        # implicit test of save_trajectory()
        assert isfile(join(path, "observations_0.pt"))
        assert buffer._n_fills == 1
        buffer.clean()
        assert not isfile(join(path, "copy_0", "log.blockMesh"))

    def test_timeout(self, temp_training):
        path, base_env = temp_training
        base_env.start_time = 0.0
        base_env.end_time = 0.015
        buffer = LocalBuffer(path, base_env, 2, 2, timeout=1)
        assert not isfile(join(path, base_env.path, "log.pimpleFoam.pre"))
