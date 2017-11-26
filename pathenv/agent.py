import collections
import random

import numpy

MemoryRecord = collections.namedtuple('MemoryRecord',
                                      'observation action reward next_observation done')


class BaseAgent(object):
    def __init__(self,
                 input_shape=None,
                 number_of_actions=1,
                 max_memory_size=250):
        self.input_shape = input_shape
        self.number_of_actions = number_of_actions
        self.max_memory_size = max_memory_size

        self.goal = None
        self.memory = []
        
        self._build_model()

    def __repr__(self):
        return self.__class__.__name__

    def _build_model(self):
        pass

    def new_episode(self, goal):
        self.memory.append([])
        self.memory = self.memory[-self.max_memory_size:]
        self.goal = goal

    def act(self, observation):
        action = numpy.random.choice(self.number_of_actions)
        
        return action

    def train_on_memory(self):
        pass

    def update_memory(self, observation, action, reward, next_observation, done):
        self.memory[-1].append(MemoryRecord(observation, action, reward, next_observation, done))

class StuntAgent(BaseAgent):
    def act(self, observation):
        action = numpy.random.choice(self.number_of_actions)
        if len(self.memory) > 0 and len(self.memory[-1]) > 0:
            reward = self.memory[-1][-1].reward
            lastaction = self.memory[-1][-1].action
            if reward > 0:
                action = numpy.random.choice(self.number_of_actions*2)
                return action if action < self.number_of_actions else lastaction
            else:
                if reward < 0 and action == lastaction:
                    action = numpy.random.choice(self.number_of_actions)
        return action

class FoxAgent(BaseAgent):
#Don't work because only one value > 0
    def act(self, observation):
        ci = observation.shape[1] // 2
        cj = observation.shape[2] // 2
        action = -1
        loopflag = (len(self.memory) == 0 or len(self.memory[-1]) == 0
                or self.memory[-1][-1].reward > - 10)
        # print(loopflag)
        while True:
            if (action == 0):
                observation[0,:ci,cj] = 0
            elif (action == 1):
                observation[0,:ci,cj+1:] = 0
            elif (action == 2):
                observation[0,ci,cj+1:] = 0
            elif (action == 3):
                observation[0,ci+1:,cj+1:] = 0
            elif (action == 4):
                observation[0,ci+1:,cj] = 0
            elif (action == 5):
                observation[0,ci+1:,:cj] = 0
            elif (action == 6):
                observation[0,ci,:cj] = 0
            elif (action == 7):
                observation[0,:ci,:cj] = 0
            else:
                observation[0,ci,cj] = 0
            # print(observation)
            hot = numpy.argmax(observation)
            (hoti, hotj) = divmod(hot, observation.shape[2])
            if (hoti == ci):
                action = 6 if hotj < cj else 2
            if (hotj == cj):
                action = 0 if hoti < ci else 4
            if (hoti < ci):
                action = 7 if hotj < cj else 1
            else:
                action = 5 if hotj < cj else 3
            # print(action)
            if loopflag or action != self.memory[-1][-1].action:
                return action    
            
class QAgent(BaseAgent):    
    def _build_model(self):
        self.table = dict()
        self.learning_factor = 0.9
        self.discount_factor = 0.9
        self.dejavu_threshold = 20
       
    def train_on_memory(self):
        for observation, action, reward, next_observation, done in reversed(self.memory[-1]):
            # Do it in a service loop
            # bins = np.arange(-1, np.max(observation) + self.splitter, self.splitter)
            # digO = bins[numpy.digitize(observation, bins) // 2 * 2]
            # bins = np.arange(-1, np.max(next_observation) + self.splitter, self.splitter)
            # digNO = bins[numpy.digitize(next_observation, bins) // 2 * 2]
            
            digO = tuple(map(tuple, observation[0,:,:]))
            digNO = tuple(map(tuple, next_observation[0,:,:]))
            if digO not in self.table:
                self.table[digO] = numpy.zeros(self.number_of_actions)
            if digNO in self.table:
                self.table[digO][action] += self.learning_factor * (reward
                                                              - self.table[digO][action] +
                                       self.discount_factor * numpy.max(self.table[digNO]))
            else:
                self.table[digO][action] += self.learning_factor * (reward
                                                              - self.table[digO][action])
    def find_cycle(self, digO):
        shift = 0
        if len(self.memory[-1]) < 3:
            return False
        if (digO != self.memory[-1][-1][0]).any():
            for i in xrange(2, min(len(self.memory[-1]) + 1, self.dejavu_threshold)):
                if (digO == self.memory[-1][-i][0]).all():
                    shift = i
                    break
            if shift == 0 or shift * 2 - 1 > len(self.memory[-1]):
                return False
            for i in xrange(shift + 1, shift * 2):
                if (self.memory[-1][-i+shift][0] != self.memory[-1][-i][0]).any():
                    return False
            return True
        else:
            return False
    
    def act(self, observation):
        digO = tuple(map(tuple, observation[0,:,:]))
        if digO not in self.table or self.find_cycle(digO):
                return numpy.random.choice(self.number_of_actions)
        return numpy.argmax(self.table[digO])
