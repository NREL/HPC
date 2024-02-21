"""This is a very simple version of the car passing problem."""

__author__ = "Xiangyu Zhang"
__email__ = "Xiangyu.Zhang@nrel.gov"

import gymnasium as gym
from gymnasium import spaces
import numpy as np

TIME_LAPSE_REWARD = -1.0
COLLISION_OOB_REWARD = -100.0
TARGET_REACHED_REWARD = 200.0
MAX_STEP_CNT = 50
DEG2RAD = np.pi / 180


class CarPassEnv(gym.Env):
    """ This is a toy car passing problem: one car stays still in front of the
        other. The agent will control the hind car to pass the stationary car 
        and eventually reach to a point directly in front of the stationary
        car. No physics such as inertia is considered (but can be easily
        added). A car is represented using a circle, the moving car has radius 
        of 1.5 and the stationary car has radius between 0.8 and 1.8 (the size 
        of the car changes). Two car collide into each other when the circle 
        overlaps. The coordinate of the destination is fixed at (2, 5) while 
        the center of the stationary car is at (2, 0).

        Observation:
        Type: Box(2)
        Num   Observation     Min     Max
        0     Position X      -4      4
        1     Position Y      -15     6

        Actions:
        Type: Box(2)
        Num   Variable        Min     Max
        0     Speed           0       3.0
        1     Direction       0       360.0

        Rewards:
        Type                  Value       Description
        Time lapse            -1          Represent the elapse of time, so the 
                                          task will be finished asap.
        Collision/OOB         -100        Two cars collide or the moving car is
                                          out of boundary, both end the game.
        Reached Destimation   200         The moving car arrives at the 
                                          designated spot.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        
        self.target_position = [2.0, 5.0]
        self.stationary_car_poistion = [2.0, 0.0]
        self.moving_car_size = 1.5
        self.stationary_car_size = None  # will be initialized in environment.

        self.state_upper_limit = np.array([4.0, 6.0])
        self.state_lower_limit = np.array([-4.0, -14.0])

        self.action_upper_limit = np.array([1.0, 1.0])
        self.action_lower_limit = np.array([-1.0, -1.0])

        self.action_space = spaces.Box(self.action_lower_limit, 
                                       self.action_upper_limit, 
                                       dtype=np.float32)
        self.observation_space = spaces.Box(self.state_lower_limit, 
                                            self.state_upper_limit, 
                                            dtype=np.float32)

        self.state = None
        self.viewer = None
        self.step_count = 0

    def step(self, action):
        """ Makes one move.

        There are three steps:
        Step 1: Calculate next position of the car.
        Step 2: Check if collision or out of boundary happens.
        Step 3: Check if target reached.

        There are three situations that ends the episode:
        1. Two cars collide.
        2. Moving car goes out of boundary.
        3. Step count reaches 50.

        Args:
          action: A Numpy array: the first element is the speed and the second 
            is direction in degree. Assuming the speed is normalized (from -1.0
            to 1.0), action speed in the environment is from 0.0 to 3.0.
        
        Returns:
          state: A 2 Numpy array: the X-Y coordinates of the moving car.
          reward: A float, numeric reward for the action taken.
          terminated: A Boolean, indicating whether the episode has ended.
          truncated: A Boolean, indicating if the episode has reached a maximum
            length. See the following link for differences between terminated
            and truncated: https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
          info: A dictionary.
        """

        self.step_count += 1
        truncated = False
        terminated = False

        # Actions in this game is bounded. 
        norm_speed = np.clip(action[0], -1.0, 1.0)  # only clip the speed.

        speed = norm_speed * 1.5 + 1.5  # recover speed from normalized value.
        direction = (action[1] * 180.0 + 180.0) * DEG2RAD
        next_position = (self.state 
                         + np.array([speed * np.sin(direction),
                                     speed * np.cos(direction)]))

        forward_move = (self.get_distance(self.state, 
                                          self.target_position) 
                        - self.get_distance(next_position, 
                                            self.target_position))

        self.state = next_position

        two_cars_dist = self.get_distance(next_position,
                                          self.stationary_car_poistion)
        
        if (two_cars_dist <= self.stationary_car_size + self.moving_car_size):
            # Collide with stationary car
            reward = COLLISION_OOB_REWARD
            terminated = True
        elif self.oob_detect(next_position):
            # out of bound
            reward = COLLISION_OOB_REWARD
            terminated = True
        elif self.get_distance(next_position, self.target_position) < 0.05:
            # Reached target
            reward = TARGET_REACHED_REWARD
            terminated = True
        else:
            # Went somewhere.
            if forward_move > 0.0:
                # This reward serves as a guidance, without it we will 
                reward = forward_move
            else:
                # if moving backwards, penalize at least a TIME_LAPSE_REWARD 
                # for wasting time.
                reward = min(forward_move, TIME_LAPSE_REWARD)

        # Need to clip it, otherwise, the observation is out of the legal
        # bound, and RLLIB will complain.
        self.state = np.clip(next_position, 
                             self.state_lower_limit, 
                             self.state_upper_limit)

        if self.step_count >= MAX_STEP_CNT:
            truncated = True

        # Info is to pass some variables' values out, but we do not use it
        # here.
        info = {}

        return self.state, reward, terminated, truncated, info

    def oob_detect(self, position):
        """ Vehicle out of bound detection.
        """
        oob_flag = (position[0] > self.state_upper_limit[0] 
                    or position[1] > self.state_upper_limit[1] 
                    or position[0] < self.state_lower_limit[0] 
                    or position[1] < self.state_lower_limit[1])
        
        return oob_flag

    def get_distance(self, coord1, coord2):
        """ Calculates the distance between two coordinates.
        
        Args:
          coord1: A list with two float numbers, represents the X-Y coordinate.
          coord2: A list with two float numbers, represents the X-Y coordinate.
        
        Returns:
          dist: A float representing the distance between these coordinates.
        """

        dist = np.sqrt((coord1[0] - coord2[0]) ** 2 
                       + (coord1[1] - coord2[1]) ** 2)
        
        return dist

    def reset(self, seed=None, options={}):

        super().reset(seed=seed)

        if options is None:
            options = {}

        self.state = np.array([2.0, np.random.uniform(-12.0, -8.0)])
        # Choose the size of the car parked ahead.
        self.stationary_car_size = np.random.uniform(0.8, 1.8)
        self.step_count = 0

        info = {}
        
        return self.state, info

    def render(self, mode='human'):
        """ Rendering visually.

        This function is totally optional. No need to render if your 
        environment has nothing to show visually.
        """

        screen_width = 240
        screen_height = 600

        world_width = 8.0  # Display from -4 to 4.
        scale = screen_width / world_width

        def cord_transform(position, scale):
            x = (position[0] + 4.0) * scale
            y = (position[1] + 14.0) * scale
            return [x, y]

        if self.viewer is None:
            from gymnasium.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            self.moving_car = rendering.make_circle(1.5 * scale)
            self.cartrans = rendering.Transform()
            self.moving_car.add_attr(self.cartrans)
            self.moving_car.set_color(0.0, 0.0, 0.0)
            self.cartrans.set_translation(*cord_transform(self.state, scale))
            self.viewer.add_geom(self.moving_car)

            self.stationary_car = rendering.make_circle(
                self.stationary_car_size * scale)
            self.stationary_car.set_color(.8, .6, .4)
            self.stationary_car.add_attr(
                rendering.Transform(
                    translation=tuple(
                        cord_transform(self.stationary_car_poistion, scale))))
            self.viewer.add_geom(self.stationary_car)

            self.target_spot = rendering.make_circle(0.5 * scale)
            self.target_spot.set_color(1.0, 0.0, 0.0)
            self.target_spot.add_attr(
                rendering.Transform(translation=tuple(
                    cord_transform(self.target_position, scale))))
            self.viewer.add_geom(self.target_spot)

        if self.state is None:
            return None

        self.cartrans.set_translation(*cord_transform(self.state, scale))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


if __name__ == "__main__":

    env = CarPassEnv()

    obs = env.reset()
    done = False
    episodic_reward = 0.0

    while not done:
        act = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(act)
        done = (terminated or truncated)
        episodic_reward += reward

    print("Reward this episode is %f" % episodic_reward)
    print("Steps this episode is %d" % env.step_count)

