import gym
import numpy
import gym.spaces
from scipy.spatial.distance import cityblock as dist_metric
import random
from .utils_compiled import build_distance_map, get_flat_state, check_finish_achievable
from gym.envs.classic_control import rendering

from .tasks import BY_PIXEL_ACTIONS, BY_PIXEL_ACTION_DIFFS, TaskSet


class PathFindingByPixelWithDistanceMapEnv(gym.Env):
    metadata = {
        'render.modes': ['rgb_array'],
        'video.frames_per_second' : 3
    }
    def __init__(self):
        self.task_set = None
        self.cur_task = None
        self.observation_space = None
        self.obstacle_punishment = None
        self.local_goal_reward = None
        self.done_reward = None

        self.distance_map = None

        self.action_space = gym.spaces.Discrete(len(BY_PIXEL_ACTIONS))
        self.cur_position_discrete = None
        self.goal_error = None
        
        self.viewer = None
        self.agent_rotation_ratio = 15
        self.steps = 0.
        self.last_agent = None
        self.trace = 'long'

    def _configure(self,
                   tasks_dir='data/imported/paths',
                   maps_dir='data/imported/maps',
                   obstacle_punishment=10,
                   local_goal_reward=5,
                   done_reward=10,
                   greedy_distance_reward_weight=0.1,
                   absolute_distance_reward_weight=0.1,
                   vision_range=20,
                   target_on_border_reward=5,
                   absolute_distance_observation_weight=0.1,
                   max_steps=100,
                   video_frame_per_second=3,
                   agent_trace = 'long' #for rendering
                  ):

        self.task_set = TaskSet(tasks_dir, maps_dir)
        self.task_ids = list(self.task_set.keys())
        self.cur_task_i = 0
        self.vision_range = vision_range

        self.observation_space = gym.spaces.Box(low=0,
                                                high=1,
                                                shape=(2 * self.vision_range + 1, 2 * self.vision_range + 1))

        self.obstacle_punishment = abs(obstacle_punishment)
        self.local_goal_reward = local_goal_reward
        self.done_reward = done_reward

        self.greedy_distance_reward_weight = greedy_distance_reward_weight
        self.absolute_distance_reward_weight = absolute_distance_reward_weight

        self.target_on_border_reward = target_on_border_reward
        self.absolute_distance_observation_weight = absolute_distance_observation_weight
        self.max_steps = max_steps
        self.steps = 0.
        self.trace = agent_trace
        self.metadata['video.frames_per_second'] = video_frame_per_second

    def __repr__(self):
        return self.__class__.__name__

    def _reset(self):
        self.steps = 0.
        self.cur_task = self.task_set[self.task_ids[self.cur_task_i]]
        self.cur_task_i += 1
        if self.cur_task_i >= len(self.task_ids):
            self.cur_task_i = 0

        rand = random.Random()
        if self.cur_task is not None:
            local_map = self.cur_task.local_map  # shortcut
            while True:
                self.start = (rand.randint(0, self.cur_task.local_map.shape[0] - 1),
                              rand.randint(0, self.cur_task.local_map.shape[1] - 1))
                self.finish = (rand.randint(0, self.cur_task.local_map.shape[0] - 1),
                               rand.randint(0, self.cur_task.local_map.shape[1] - 1))
                cstart = (self.start[1], self.start[0])
                cfinish = (self.finish[1], self.finish[0])
                if local_map[cstart] == 0 \
                        and local_map[cfinish] == 0 \
                        and cstart != cfinish \
                        and check_finish_achievable(numpy.array(local_map, dtype=numpy.float),
                                                    numpy.array(cstart, dtype=numpy.int),
                                                    numpy.array(cfinish, dtype=numpy.int)):
                    break

        return self._init_state()

    def _init_state(self):
        local_map = numpy.array(self.cur_task.local_map, dtype=numpy.float)
        self.distance_map = build_distance_map(local_map,
                                               numpy.array(self.finish, dtype=numpy.int))

        m = self.cur_task.local_map
        self.obstacle_points_for_vis = [(x, y)
                                        for y in xrange(m.shape[0])
                                        for x in xrange(m.shape[1])
                                        if m[y, x] > 0]
        self.cur_episode_state_id_seq = [tuple(self.start)]
        self.cur_position_discrete = self.start
        return self._get_state()

    def _get_base_state(self, cur_position_discrete):
        return get_flat_state(self.cur_task.local_map,
                              tuple(cur_position_discrete),
                              self.vision_range,
                              self.done_reward,
                              self.target_on_border_reward,
                              self.start,
                              self.finish,
                              self.absolute_distance_observation_weight)

    def _get_state(self):
        cur_pos = tuple(self.cur_position_discrete)
        if cur_pos != self.cur_episode_state_id_seq[-1]:
            self.cur_episode_state_id_seq.append(cur_pos)
        result = [self._get_base_state(pos)
                  for pos in self.cur_episode_state_id_seq[:-2:-1]]
        if len(result) < 1:
            empty = numpy.zeros_like(result[0])
            for _ in xrange(1 - len(result)):
                result.append(empty)
        return numpy.stack(result)

    def _step(self, action):
        self.steps = self.steps + 1
        new_position = self.cur_position_discrete + BY_PIXEL_ACTION_DIFFS[action]

        done = numpy.allclose(new_position, self.finish)
        if done:
            reward = self.done_reward
        else:
            goes_out_of_field = any(new_position < 0) or any(new_position + 1 > self.cur_task.local_map.shape)
            invalid_step = goes_out_of_field or tuple(new_position) in self.obstacle_points_for_vis
            if invalid_step:
                '''
                print("invalid")
                print(tuple(self.cur_position_discrete))
                print(new_position)
                print(any(new_position < 0))
                print(any(new_position + 1 > self.cur_task.local_map.shape))
                print(self.cur_task.local_map[tuple(new_position)] > 0)
                '''
                reward = -self.obstacle_punishment
            else:
                '''
                print("correct")
                print(tuple(self.cur_position_discrete))
                print(new_position)
                print(any(new_position < 0))
                print(any(new_position + 1 > self.cur_task.local_map.shape))
                print(self.cur_task.local_map[tuple(new_position)] > 0)
                '''
                local_target = self.finish
                cur_target_dist = dist_metric(new_position, local_target)
                if cur_target_dist < 1:
                    reward = self.local_goal_reward
                    done = True
                else:
                    reward = self._get_usual_reward(self.cur_position_discrete, new_position)
                self.cur_position_discrete = self.cur_position_discrete + BY_PIXEL_ACTION_DIFFS[action]

        observation = self._get_state()
        return observation, reward, done, None

    def _get_usual_reward(self, old_position, new_position):
        old_height = self.distance_map[tuple(old_position)]
        new_height = self.distance_map[tuple(new_position)]
        true_gain = old_height - new_height
        #print(true_gain)
        local_target = self.finish
        old_dist = dist_metric(old_position, local_target)
        new_dist = dist_metric(new_position, local_target)
        greedy_gain = old_dist - new_dist
        #print(greedy_gain)
        start_height = self.distance_map[tuple(self.start)]
        abs_gain = numpy.exp(-new_height / start_height)
        #print(abs_gain)
        total_gain = sum(
            ((1 - self.greedy_distance_reward_weight - self.absolute_distance_reward_weight) * true_gain,
             self.greedy_distance_reward_weight * greedy_gain,
             self.absolute_distance_reward_weight * abs_gain))
        #print(total_gain)
        return total_gain
    
    def _render(self, mode='rgb_array', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        #object const
        m = self.cur_task.local_map
        cwidth = screen_width / m.shape[1]
        cheight = screen_height / m.shape[0]
        csize = min(cwidth, cheight) #cell size
        screen_width = csize * m.shape[1]
        screen_height = csize * m.shape[0]
        
        if self.steps < 0.1:
            self.viewer = None
        if self.viewer is None:
            #drawlib
            self.viewer = rendering.Viewer(screen_width, screen_height)
            '''
            #Also there is set_bound()
            
            hborder = rendering.Line((0, csize * m.shape[0]), (csize * m.shape[1], csize * m.shape[0]))
            hborder.set_color(0,0,0)
            self.viewer.add_geom(hborder)
            vborder = rendering.Line((csize * m.shape[1], 0), (csize * m.shape[1], csize * m.shape[0]))
            vborder.set_color(0,0,0)
            self.viewer.add_geom(vborder)
            '''
            
            #for y in xrange(m.shape[0]):
            #    for x in xrange(m.shape[1]):
            #        if m[y, x] > 0:
            for (x, y) in self.obstacle_points_for_vis:
                        cell = rendering.FilledPolygon([(x*csize,y*csize)
                                                        , (x*csize,(y+1)*csize)
                                                        , ((x+1)*csize, (y+1)*csize)
                                                        , ((x+1)*csize,y*csize)])
                        cell.set_color(0,0,0)
                        self.viewer.add_geom(cell)
                        
        start = rendering.make_circle(csize/2, filled=False)
        start.add_attr(rendering.Transform(translation=(self.start[0] * csize + csize/2
                                                            , self.start[1] * csize + csize/2)))
        start.set_color(0,1,0)
        self.viewer.add_onetime(start)
        
        finish = rendering.make_circle(csize/2, filled=False)
        finish.add_attr(rendering.Transform(translation=(self.finish[0] * csize + csize/2
                                                            , self.finish[1] * csize + csize/2)))
        finish.set_color(1,0,0)
        self.viewer.add_onetime(finish)
        
            
        if not self.cur_position_discrete is None:
            (x, y) = tuple(self.cur_position_discrete)
            agent = rendering.make_polyline([(0, -csize / 2)
                                             #, (0, csize / 2) #let it be T
                                             , (0, 0)
                                             , (-csize / 2, 0)
                                             , (csize / 2, 0)])
            if self.trace == 'long':
                agent.set_color(0, self.steps/self.max_steps,1) 
                agent.add_attr(rendering.Transform(translation=(x * csize + csize / 2, y * csize + csize / 2)
                                              ,rotation=(self.steps%self.agent_rotation_ratio)*360//self.agent_rotation_ratio))
                self.viewer.add_geom(agent)
            else:
                if self.trace == 'short' and self.last_agent != None:
                    self.last_agent.set_color(0, 1, 1)
                    self.viewer.add_onetime(self.last_agent)
                agent.set_color(0, 0, 1)
                self.viewer.add_onetime(agent)
                self.last_agent = agent
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
    
    
    def show_map(self):
        print(self.cur_task.local_map)