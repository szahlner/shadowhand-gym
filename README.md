# shadowhand-gym

OpenaAI Gym Shadow Dexterous Hand robot environment based on PyBullet.  
Successor of the [old Shadow Dexterous Hand robot gym environment](https://rgit.acin.tuwien.ac.at/matthias.hirschmanner/shadow_teleop/-/tree/master/gym_environments).

[![GitHub](https://img.shields.io/github/license/szahlner/shadowhand-gym.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Installation

### From source

```bash
git clone https://github.com/szahlner/shadowhand-gym.git
pip install -e shadowhand-gym
```

## Usage

```python
import gym
import shadowhand_gym

env = gym.make('ShadowHandReach-v1', render=True)

obs = env.reset()
done = False
while not done:
    # Random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

env.close()
```

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/szahlner/shadowhand-gym/blob/master/examples/ShadowHandReach-v1_Example.ipynb)

## Environments

| | |
| :------------------------------: | :--------------------------------------------: |
| `ShadowHandReach-v1` **easy mode** | `ShadowHandReach-v1` **hard mode** |
| ![ShadowHandReach-v1 easy](https://raw.githubusercontent.com/szahlner/shadowhand-gym/master/docs/ShadowHandReach-v1_easy.gif) | ![ShadowHandReach-v1_hard](https://raw.githubusercontent.com/szahlner/shadowhand-gym/master/docs/ShadowHandReach-v1_hard.gif) |
| `ShadowHandBlock-v1` **orientation only** | |
| ![ShadowHandBlock-v1](https://raw.githubusercontent.com/szahlner/shadowhand-gym/master/docs/ShadowHandBlock-v1.gif) | |

Environments are widely inspired from [OpenAI ShadowHand environments](https://openai.com/blog/ingredients-for-robotics-research/).  
Code is inspired from [qgallouedec's panda-gym](https://github.com/qgallouedec/panda-gym).