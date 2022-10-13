# Deep Reinforcement Learning with OpenFOAM

## Overview

### Repository structure

- **docs**: Sphinx documentation sources (TODO)
- **drlfoam**: Python library for DRL with OpenFOAM
- **examples**: annotated scripts for performing DRL trainings and tests
- **openfoam**: OpenFOAM simulation setups (*test_cases*) and source files for additional OpenFOAM library components, e.g., boundary conditions (*src*)

### drlFoam package

- **drlfoam.environment**: wrapper classes for manipulating and parsing OpenFOAM test cases
- **drlfoam.agent**: policy and value networks, agent logic and training
- **drlfoam.execution**: trajectory buffer, local and cluster (Slurm) execution

## Installation

### Python environment

A miss-match between the installed Python-frontend of PyTorch and the C++ frontend (libTorch) can lead to unexpected behavior. Therefore, it is recommended to create a virtual environment using [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/en/latest/miniconda.html). Moreover, the currently used PyTorch version (1.12.1) requires a Python interpreter **>=3.8**. The following listing shows all steps to set up a suitable virtual environment on Ubuntu 20.04 or newer.
```
# install venv
sudo apt update && sudo apt install python3.8-venv
# if not yet cloned, get drlfoam
git clone https://github.com/OFDataCommittee/drlfoam.git
cd drlfoam
# create a new environment called pydrl
python3 -m venv pydrl
# start the environment and install dependencies
source pydrl/bin/activate
pip install -r requirements.txt
# once the work is done, leave the environment
deactivate
```

### OpenFOAM library components

Source code and test cases are only compatible with **OpenFOAM-v2206**; [installation instructions](https://develop.openfoam.com/Development/openfoam/-/wikis/precompiled). You can use the pre-compiled binaries. Make sure that the OpenFOAM environment variables are available, e.g., by adding `source /usr/lib/openfoam/openfoam2206/etc/bashrc` to `$HOME/.bashrc`. The additional OpenFOAM library components are compiled as follows:
```
# at the top-level of this repository
# when executing for the first time, libTorch is downloaded by Allwmake
source setup-env
./Allwmake
```
In case you want to re-compile starting from a clean state, remove the library folder:
```
# assuming you are at the repository's top folder
rm -r openfoam/libs/
```

### Working with Singularity containers

Instead of installing dependencies manually, you can also employ the provided Singularity container. Singularity simplifies execution on HPC infrastructures, because no dependencies except for Singularity itself and OpenMPI are required. To build the container locally, run:
```
sudo singularity build of2206-py1.11.0-cpu.sif docker://andreweiner/of_pytorch:of2206-py1.11.0-cpu
```
By default, the container is expected to be located at the repository's top level. The default location may be changed by adjusting the `DRL_IMAGE` variable in *setup-env*. To build the OpenFOAM library components, provide the `--container` flag:
```
./Allwmake --container
```

## Running a training

Currently, there is only one example for assembling a DRL training with drlFoam using the *rotatingCylinder* test case. To perform the training locally, execute the following steps:
```
# from the top-level of this repository
source pydrl/bin/activate
source setup-env
cd examples
# see run_trajectory.py for all available options
# training saved in test_training; buffer size 4; 2 runners
# this training requires 4 MPI ranks on average and two loops
# of each runner to fill the buffer
python3 run_training.py -o test_training -b 4 -r 2
```
To run the training with the Singularity container, pass the `--container` flag to *setup-env*:
```
source setup-env --container
python3 run_training.py -o test_training -b 4 -r 2
```

## Development

Unittests are implemented with [PyTest](https://docs.pytest.org/en/7.1.x/). Some tests require a [Slurm](https://slurm.schedmd.com/documentation.html) installation on the test machine. Instructions for a minimal Slurm setup on Ubuntu are available [here](https://gist.github.com/ckandoth/2acef6310041244a690e4c08d2610423). If Slurm is not available, related tests are ignored. Some test require additional test data, which can be created with the *create_test_data* script.
```
# examples for running all or selected tests
# starting from the repository top-level
# run all available tests with additional output (-s)
pytest -s drlfoam
# run all tests in the agent sub-package
pytest -s drlfoam/agent
# run all tests for the ppo_agent.py module
pytest -s drlfoam/agent/tests/test_ppo_agent.py
```

## Contributions

drlFoam is currently developed and maintained by Andre Weiner (@AndreWeiner). Significant contributions to usability, bug fixes, and tests have been made during the first [OpenFOAM-Machine Learning Hackathon](https://github.com/OFDataCommittee/OFMLHackathon) by (in alphabetical order):

- Ajay Navilarekal Rajgopal (@ajaynr)
- Darshan Thummar (@darshan315)
- Guilherme Lindner (@guilindner)
- Julian Bissantz (@jbissantz)
- Morgan Kerhouant
- Mosayeb Shams (@mosayebshams)
- Tomislav Marić (@tmaric)

The foundation of the drlFoam implementation are the student projects by [Darshan Thummar](https://github.com/darshan315/flow_past_cylinder_by_DRL) and [Fabian Gabriel](https://github.com/FabianGabriel/Active_flow_control_past_cylinder_using_DRL).

## License

drlFoam is [GPLv3](https://en.wikipedia.org/wiki/GNU_General_Public_License)-licensed; refer to the [LICENSE](https://github.com/OFDataCommittee/drlfoam/blob/main/LICENSE) file for more information.