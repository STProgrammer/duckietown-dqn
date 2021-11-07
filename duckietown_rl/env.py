import gym

from wrappers import NormalizeWrapper, ImgWrapper, DtRewardWrapper, ActionWrapper, ResizeWrapper, DiscreteWrapper

def launch_env(id=None):
    env = None
    if id is None:
        from gym_duckietown.simulator import Simulator
        env = Simulator(
            seed=2502,  # random seed
            map_name="loop_empty",
            max_steps=500001,  # we don't want the gym to reset itself
            domain_rand=False,
            camera_width=640,
            camera_height=480,
            accept_start_angle_deg=4,  # start close to straight
            full_transparency=True,
            distortion=True,
            user_tile_start=0,
        )
    else:
        env = gym.make(id)

    # Wrappers
    env = ResizeWrapper(env, (80,60,3))
    env = ImgWrapper(env)  # to make the images from HxWxC into CxHxW for PyTorch
    env = NormalizeWrapper(env)
    env = ActionWrapper(env)
    env = DiscreteWrapper(env)
    env = DtRewardWrapper(env)

    return env