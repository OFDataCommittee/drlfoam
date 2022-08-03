
from os import environ
from os.path import join
from torch import DoubleTensor


DEFAULT_TENSOR_TYPE = DoubleTensor
EPS_SP = 1.0e-6
EPS_DP = 1.0e-14
BASE_PATH = environ.get("DRL_BASE", "")
TESTCASE_PATH = join(BASE_PATH, "openfoam", "test_cases")
TESTDATA_PATH = join(BASE_PATH, "test_data")