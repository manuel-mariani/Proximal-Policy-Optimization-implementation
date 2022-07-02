# Proximal Policy Optimization
Project work for Autonomous and Adaptive Systems course, UNIBO 2022.

The project consists of an implementation of PPO from scratch, using various algorithmic improvements to boost its 
performance. As a baseline to evaluate the algorithm, a simple REINFORCE algorithm is also implemented.

As a benchmark, the CoinRun environment from [procgen](https://github.com/openai/procgen) is used.
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

## Results
Traning for less than 1h on a single GPU:
- Using REINFORCE, agent reaches 71% win rate in an average of 90 steps per episode. [(wandb)](https://wandb.ai/mmariani/AAS-RL/reports/REINFORCE-Nature-CNN--VmlldzoyMjU3NjQ1?accessToken=vhto55xlgp9ron077811ugfbn2y0i1fpkvnc1sh7b662dyjj1lkagdvujs0r1h5y)
- Using PPO, agent reaches 78% win rate in an average of 55 steps per episode [(wandb)](https://wandb.ai/mmariani/AAS-RL/reports/PPO-Nature-CNN--VmlldzoyMjU3Njcy?accessToken=lmzq45x20grozbpq23ngq9to75hyzy2nw3lop8hd9qcuudqorzx3qkdurb3ide3p)