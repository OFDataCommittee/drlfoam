
from os import remove
from os.path import join, isfile, isdir
from re import findall
from subprocess import Popen, PIPE
from shutil import which
import pytest
from .test_local import temp_training
from ..slurm import (SlurmBuffer, SlurmConfig, submit_job,
                     get_job_status, submit_and_wait)


slurm_available = pytest.mark.skipif(
    which("sbatch") is None, reason="Slurm workload manager not available"
)


@slurm_available
def test_submit_job():
    config = SlurmConfig(
        commands=["sleep 1"], job_name="testjob", n_tasks=1, mem_per_cpu=1
    )
    path = join("/tmp", "testjob.sh")
    config.write(path)
    job_id = submit_job(path)
    assert isinstance(job_id, int)
    if isfile(path):
        remove(path)


@slurm_available
def test_get_job_status():
    config = SlurmConfig(
        commands=["sleep 1"], job_name="testjob", n_tasks=1, mem_per_cpu=1
    )
    path = join("/tmp", "testjob.sh")
    config.write(path)
    job_id = submit_job(path)
    status = get_job_status(job_id)
    assert status in ["R", "CF", "PD"]
    if isfile(path):
        remove(path)
    with pytest.raises(Exception):
        get_job_status(0)


@slurm_available
def test_submit_and_wait():
    config = SlurmConfig(
        commands=["sleep 2"], job_name="testjob", n_tasks=1, mem_per_cpu=1
    )
    path = join("/tmp", "testjob.sh")
    config.write(path)
    submit_and_wait(path, 1)
    assert True


class TestSlurmConfig():
    def test_empty_write(self):
        config = SlurmConfig()
        path = join("/tmp", "emptyjob.sh")
        config.write(path)
        assert isfile(path)
        with open(path, "r") as job:
            content = job.read()
            assert "bash" in content
        remove(path)

    def test_write_options(self):
        config = SlurmConfig(
            job_name="testjob",
            n_tasks=8,
            mail_user="name@mail.com",
            n_nodes=1,
            time="120:00:00",
            partition="debug"
        )
        path = join("/tmp", "testjob.sh")
        config.write(path)
        assert isfile(path)
        with open(path, "r") as job:
            content = job.read()
            assert "testjob" in content
            assert "name@mail.com" in content
            assert "120:00:00" in content
            assert "debug" in content
        remove(path)

    def test_write_modules(self):
        modules = [
            "singularity/3.6.0rc2",
            "mpi/openmpi/4.1.1/gcc"
        ]
        config = SlurmConfig(job_name="testjob")
        config.modules = modules
        path = join("/tmp", "testjob.sh")
        config.write(path)
        assert isfile(path)
        with open(path, "r") as job:
            content = job.read()
            for m in modules:
                assert f"module load {m}" in content
        remove(path)

    def test_write_commands(self):
        modules = [
            "singularity/3.6.0rc2",
            "mpi/openmpi/4.1.1/gcc"
        ]
        commands_pre = [
            "source environment.sh",
            "source OpenFOAM commands"
        ]
        commands = [
            "cd where/to/run",
            "./Allrun"
        ]
        config = SlurmConfig(job_name="testjob")
        config.modules = modules
        config.commands_pre = commands_pre
        config.commands = commands
        path = join("/tmp", "testjob.sh")
        config.write(path)
        assert isfile(path)
        with open(path, "r") as job:
            content = job.read()
            for c in commands_pre + commands:
                assert c in content
        remove(path)


@slurm_available
class TestSlurmBuffer():
    def test_prepare_reset_fill_clean(self, temp_training):
        path, base_env = temp_training
        base_env.start_time = 0.0
        base_env.end_time = 0.015
        config = SlurmConfig(n_tasks=2, mem_per_cpu=2)
        buffer = SlurmBuffer(path, base_env, 2, 2, config)
        buffer.prepare()
        base_env.end_time = 0.03
        assert isfile(join(path, base_env.path, "log.blockMesh"))
        assert isfile(join(path, base_env.path, "trajectory.csv"))
        assert isdir(join(path, base_env.path, "postProcessing"))
        envs = buffer.envs
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
        config = SlurmConfig(n_tasks=2, mem_per_cpu=2)
        buffer = SlurmBuffer(path, base_env, 2, 2, config, timeout=1, wait=1)
        buffer.prepare()
        assert not isfile(join(path, base_env.path, "log.pimpleFoam.pre"))
