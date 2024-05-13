import os
import random
import time
import numpy as np
import cv2
import math
import subprocess
import atexit
import signal
import carla
from ding.envs import BaseEnv, BaseEnvTimestep
import gym
import copy
from typing import Any, List, Tuple
from ding.utils import ENV_REGISTRY

@ENV_REGISTRY.register('carla_lightzero')
class CarlaEnv(BaseEnv):

    config = dict(
        steer_amt=1.0,
        im_width=640,
        im_height=480,
        seconds_per_episode=60,
        front_camera=None,
        show_cam = False
    )

    def __init__(self, cfg):
        self._cfg = cfg
        self.steer_amt = self._cfg.steer_amt
        self.im_width = self._cfg.im_width
        self.im_height = self._cfg.im_height
        self.seconds_per_episode = self._cfg.seconds_per_episode
        self.front_camera = self._cfg.front_camera
        self.show_cam = self._cfg.show_cam

        attempts = 0
        num_max_restarts = 5
        while attempts < num_max_restarts:
            port = np.random.randint(2000, 10000)
            while self.isInuse(port):
                port = np.random.randint(2000, 10000)
            cmd1 = f'./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port={port}'
            self.server = subprocess.Popen(cmd1, shell=True, preexec_fn=os.setsid)
            atexit.register(os.killpg, self.server.pid, signal.SIGKILL)
            time.sleep(10)
            try:
                self.client = carla.Client("localhost", port)
                self.client.set_timeout(30.0)
                self.world = self.client.get_world()
                break
            except:
                attempts += 1
                os.killpg(self.server.pid, signal.SIGKILL)
                atexit.unregister(lambda: os.killpg(self.server.pid, signal.SIGKILL))
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.actor_list = []

    def reset(self):
        self._destroy_agents()
        self.collision_hist = []
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        action_mask = np.ones(3, 'int8')
        observation = np.transpose(self.front_camera, (2, 0, 1))
        obs = {'observation': observation, 'action_mask': action_mask, 'to_play': -1}
        return obs

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        #print(i.shape)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.show_cam:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.steer_amt))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.steer_amt))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + self.seconds_per_episode < time.time():
            done = True
        if done:
            self._destroy_agents()
            os.killpg(self.server.pid, signal.SIGKILL)
            atexit.unregister(lambda: os.killpg(self.server.pid, signal.SIGKILL))
        
        action_mask = np.ones(3, 'int8')
        observation = np.transpose(self.front_camera, (2, 0, 1))
        obs = {'observation': observation, 'action_mask': action_mask, 'to_play': -1}
        return BaseEnvTimestep(obs, reward, done, None)
    
    def _destroy_agents(self):
        for actor in self.actor_list:
            # If it has a callback attached, remove it first
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()
            # If it's still alive - desstroy it
            if actor.is_alive:
                actor.destroy()
        self.actor_list = []
    
    def close(self):
        self._destroy_agents()
        os.killpg(self.server.pid, signal.SIGKILL)
        atexit.unregister(lambda: os.killpg(self.server.pid, signal.SIGKILL))

    def __repr__(self) -> str:
        return "DI-engine Carla Env({})".format(self._cfg.env_id)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_cfg = copy.deepcopy(cfg)
        collector_env_num = collector_cfg.pop('collector_env_num', 1)
        return [collector_cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_cfg = copy.deepcopy(cfg)
        evaluator_env_num = evaluator_cfg.pop('evaluator_env_num', 1)
        return [evaluator_cfg for _ in range(evaluator_env_num)]

    @property
    def observation_space(self) -> gym.spaces.Space:
        pass

    @property
    def action_space(self) -> gym.spaces.Space:
        pass

    @property
    def reward_space(self) -> gym.spaces.Space:
        pass

    def isInuse(self, port):
        if os.popen('netstat -na | grep :' + str(port)).readlines(): 
            portIsUse = True 
            print('%d is inuse' % port) 
        else: 
            portIsUse = False 
            print('%d is free' % port) 
        return portIsUse