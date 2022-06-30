# Proximal Policy Optimization
Project work for Autonomous and Adaptive Systems course, UNIBO 2022.

The project consists of an implementation of PPO from scratch, using various algorithmic improvements to boost its 
performance. As a baseline to evaluate the algorithm, a simple REINFORCE algorithm is also implemented.

A more detailed explanation of the project can be found in `docs/report.pdf`.


## Dependencies
Dependencies can be installed using conda.
```
conda env create -f environment.yml
# Alternative for exact dependencies
conda env create -f environment-explicit.yml
```

## Running
To test the agent, navigate to the `src` directory, activate the conda environment with `conda activate rl` and run
```
python main.py
```

To train the agent, modify the `MAIN.py` file with `TRAIN = True` in the config section.