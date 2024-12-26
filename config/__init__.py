from .gym import *
from .neorl import *


loaded_args = {
    # d4rl gym
    "halfcheetah": halfcheetah_config,
    "hopper": hopper_config,
    "walker2d": walker2d_config,

    # neorl
    "HalfCheetah-v3-low": halfcheetah_v3_low_args,
    "Hopper-v3-low": hopper_v3_low_args,
    "Walker2d-v3-low": walker2d_v3_low_args,
    "HalfCheetah-v3-medium": halfcheetah_v3_medium_args,
    "Hopper-v3-medium": hopper_v3_medium_args,
    "Walker2d-v3-medium": walker2d_v3_medium_args,
    "HalfCheetah-v3-high": halfcheetah_v3_high_args,
    "Hopper-v3-high": hopper_v3_high_args,
    "Walker2d-v3-high": walker2d_v3_high_args,
}