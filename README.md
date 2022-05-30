# RL FOCS LAB

RL FOCS LAB is an extension of optical rl gym to more realistic networking scenarios with a more accurate physical layer model.
We have retained the same code style for compatibility. 

# Installation

You can install the Optical RL-Gym with:

```bash
git clone https://github.com/carlosnatalino/optical-rl-gym.git
cd optical-rl-gym
pip install -e .
``` 
You can see the dependencies in the [setup.py](setup.py) file. These can be installed easily with pip/conda. 

We use the implementation of PPO with invalid action masking provided by the Stable-Baselines Contrib library: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib 
