"""
helper functions
"""
import sys
import logging
import fileinput

from glob import glob
from typing import Any, Union
from os.path import isdir, isfile, basename, join

logger = logging.getLogger(__name__)


def get_time_folders(path: str) -> list:
    def is_float(element: Any) -> bool:
        # taken from:
        # https://stackoverflow.com/questions/736043/checking-if-a-string-can-be-converted-to-float-in-python
        try:
            float(element)
            return True
        except ValueError:
            return False
    folders = [basename(x) for x in glob(join(path, "[0-9]*"))
               if isdir(x) and is_float(basename(x))]
    return folders


def get_latest_time(path: str) -> str:
    folders = get_time_folders(path)
    if not folders:
        if isdir(join(path, "0.org")):
            return "0.org"
        else:
            raise ValueError(f"Could not find time folder in {path}")
    return sorted(folders, key=float)[-1]


def fetch_line_from_file(path: str, keyword: str) -> str:
    with open(path) as f:
        lines = []
        for line in f.readlines():
            if keyword in line:
                lines.append(line)
        return lines if len(lines) > 1 else lines[0]


def replace_line_in_file(path: str, keyword: str, new: str) -> None:
    """Keyword-based replacement of one or more lines in a file.

    :param path: file location
    :type path: str
    :param keyword: keyword based on which lines are selected
    :type keyword: str
    :param new: the new line replacing the old one
    :type new: str
    """
    new = new + "\n" if not new.endswith("\n") else new
    fin = fileinput.input(path, inplace=True)
    for line in fin:
        if keyword in line:
            line = new
        sys.stdout.write(line)
    fin.close()


def replace_line_latest(path: str, filename: str, keyword: str, new: str,
                        processor: bool = True) -> None:
    search_path = join(path, "processor0") if processor else path
    latest_time = get_latest_time(search_path)
    if processor:
        for p in glob(join(path, "processor*")):
            replace_line_in_file(
                join(p, latest_time, filename), keyword, new
            )
    else:
        replace_line_in_file(
            join(path, latest_time, filename), keyword, new
        )


def check_path(path: str) -> None:
    if not isdir(path):
        raise ValueError(f"Could not find path {path}")


def check_file(file_path: str) -> None:
    if not isfile(file_path):
        raise ValueError(f"Could not find file {file_path}")


def check_pos_int(value: int, name: str, with_zero=False) -> None:
    message = f"Argument {name} must be a positive integer; got {value}"
    if not isinstance(value, int):
        raise ValueError(message)
    lb = 0 if with_zero else 1
    if value < lb:
        raise ValueError(message)


def check_pos_float(value: float, name: str, with_zero=False) -> None:
    message = f"Argument {name} must be a positive float; got {value}"
    if not isinstance(value, (float, int)):
        raise ValueError(message)
    if with_zero and value < 0.0:
        raise ValueError(message)
    if not with_zero and value <= 0.0:
        raise ValueError(message)


def check_finish_time(base_path: str, t_end: Union[int, float], simulation: str) -> None:
    """
    checks if the user-specified finish time is greater than the end time of the base case, if not then exit with
    an error message

    :param base_path: BASE_PATH defined in run_training
    :param t_end: user-specified finish time
    :param simulation: test case
    :return: None
    """
    pwd = join(base_path, "openfoam", "test_cases", simulation, "system", "controlDict")
    with open(pwd, "r") as f:
        lines = f.readlines()

    # get the end time of the base case, normally endTime is specified in l. 28, but in case of modifications, check
    # lines 20-35
    t_base = [float(i.strip(";\n").split(" ")[-1]) for i in lines[20:35] if i.startswith("endTime")][0]

    if t_base >= t_end:
        logger.critical(f"specified finish time is smaller than end time of base case! The finish time needs to be "
                        f"greater than {t_base}. Exiting...")
        exit(0)
