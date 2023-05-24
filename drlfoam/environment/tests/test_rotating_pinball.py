from os.path import join, exists
from shutil import copytree, rmtree
from pytest import raises, fixture
from ..rotating_pinball import (RotatingPinball2D,
                                _parse_surface_field_sum,
                                _parse_forces, _parse_probes,
                                _parse_trajectory)
from ...constants import TESTDATA_PATH
from ...utils import fetch_line_from_file


@fixture()
def temp_case():
    case = "rotatingPinball2D"
    source = join(TESTDATA_PATH, case)
    dest = join("/tmp", case)
    yield copytree(source, dest, dirs_exist_ok=True)
    rmtree(dest)


def test_parse_surface_field_sum(temp_case):
    path = join(temp_case, "postProcessing", "field_cylinder_a",
                "0", "surfaceFieldValue.dat")
    sfs = _parse_surface_field_sum(path)
    assert all(sfs.columns == ["t", "cx", "cy", "cz"])
    assert len(sfs) == 3
    assert not sfs.isnull().values.any()


def test_parse_forces(temp_case):
    forces = _parse_forces(temp_case)
    columns = ["t_a", "cx_a", "cy_a", "cz_a",
               "t_b", "cx_b", "cy_b", "cz_b",
               "t_c", "cx_c", "cy_c", "cz_c"
    ]
    assert all(forces.columns == columns)
    assert len(forces) == 3
    assert not forces.isnull().values.any()


def test_parse_probes(temp_case):
    path = join(temp_case, "postProcessing", "probes", "0", "p")
    probes = _parse_probes(path, 14)
    columns = ["t"] + [f"p{i}" for i in range(14)]
    assert all(probes.columns == columns)
    assert len(probes) == 3
    assert not probes.isnull().values.any()


def test_parse_trajectory(temp_case):
    path = join(temp_case, "trajectory.csv")
    tr = _parse_trajectory(path)
    columns = [
        "t",
        "omega_a", "alpha_a", "beta_a",
        "omega_b", "alpha_b", "beta_b",
        "omega_c", "alpha_c", "beta_c"
    ]
    assert all(tr.columns == columns)
    assert len(tr) == 3
    assert not tr.isnull().values.any()


class TestRotatingPinball(object):
    def test_common(self, temp_case):
        _ = RotatingPinball2D()
        assert True

    def test_start_time(self, temp_case):
        env = RotatingPinball2D()
        env.initialized = True
        env.path = temp_case
        env.start_time = 3.0
        lines = fetch_line_from_file(
            join(env.path, "system", "controlDict"),
            "timeStart"
        )
        assert len(lines) == 5
        assert all(["3.0" in line for line in lines])

    def test_end_time(self, temp_case):
        env = RotatingPinball2D()
        env.initialized = True
        env.path = temp_case
        env.end_time = 300.0
        line = fetch_line_from_file(
            join(env.path, "system", "controlDict"),
            "endTime "
        )
        assert "300.0" in line

    def test_control_interval(self, temp_case):
        env = RotatingPinball2D()
        env.initialized = True
        env.path = temp_case
        env.control_interval = 2.0
        lines = fetch_line_from_file(
            join(env.path, "system", "controlDict"),
            "executeInterval"
        )
        assert len(lines) == 5
        assert all(["2.0" in line for line in lines])
        lines = fetch_line_from_file(
            join(env.path, "system", "controlDict"),
            "writeInterval"
        )
        assert len(lines) == 6
        assert all(["2.0" in line for line in lines])

    def test_action_bounds(self, temp_case):
        env = RotatingPinball2D()
        env.initialized = True
        env.path = temp_case
        env.action_bounds = 10.0
        line = fetch_line_from_file(
            join(env.path, "processor1", "2", "U"),
            "absOmegaMax"
        )
        assert "10.0" in line

    def test_seed(self, temp_case):
        env = RotatingPinball2D()
        env.initialized = True
        env.path = temp_case
        env.seed = 10
        line = fetch_line_from_file(
            join(env.path, "processor2", "2", "U"),
            "seed"
        )
        assert "10" in line

    def test_policy(self, temp_case):
        env = RotatingPinball2D()
        env.initialized = True
        env.path = temp_case
        env.policy = "model.pt"
        line = fetch_line_from_file(
            join(env.path, "processor3", "2", "U"),
            "policy"
        )
        assert "model.pt" in line

    def test_train(self, temp_case):
        env = RotatingPinball2D()
        env.initialized = True
        env.path = temp_case
        env.train = False
        line = fetch_line_from_file(
            join(env.path, "processor0", "2", "U"),
            "train"
        )
        assert "false" in line

    def test_reset(self, temp_case):
        env = RotatingPinball2D()
        env.initialized = True
        env.path = temp_case
        env.start_time = 2.0
        env.reset()
        assert not exists(join(env.path, "log.pimpleFoam"))
        assert not exists(join(env.path, "trajectory.csv"))
        assert not exists(join(env.path, "postProcessing"))

    def test_observations(self, temp_case):
        env = RotatingPinball2D()
        env.initialized = True
        env.path = temp_case
        obs = env.observations
        assert len(obs.keys()) == 15
        assert all([obs[key].shape[0] == 3 for key in obs])
        assert obs["states"].shape == (3, 14)
        assert obs["actions"].shape == (3, 3)
        assert obs["rewards"].shape == (3,)

