import random
from carla_env import CarlaEnv
from easydict import EasyDict


if __name__ == '__main__':
    while True:
        print('Restarting episode')
        config = EasyDict(
            steer_amt=1.0,
            im_width=640,
            im_height=480,
            seconds_per_episode=60,
            front_camera=None,
            show_cam = False
        )

        env = CarlaEnv(config)
        current_state = env.reset()
        done = False
        
        # Loop over steps
        while True:
            action = random.randint(0, 3)
            current_state, reward, done, _ = env.step(action)
            print(current_state, reward, done)
            # If done - agent crashed, break an episode
            if done:
                break
