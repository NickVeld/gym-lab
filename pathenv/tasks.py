import cPickle
import collections
import os

import numpy

BY_PIXEL_ACTIONS = {
    0: 'N',
    1: 'NE',
    2: 'E',
    3: 'SE',
    4: 'S',
    5: 'SW',
    6: 'W',
    7: 'NW'
}
BY_PIXEL_ACTION_DIFFS = {
    0: numpy.array([-1, 0], dtype='int8'),
    1: numpy.array([-1, 1], dtype='int8'),
    2: numpy.array([0, 1], dtype='int8'),
    3: numpy.array([1, 1], dtype='int8'),
    4: numpy.array([1, 0], dtype='int8'),
    5: numpy.array([1, -1], dtype='int8'),
    6: numpy.array([0, -1], dtype='int8'),
    7: numpy.array([-1, -1], dtype='int8')
}
COMPACT_MAP_EXT = '.npz'
COMPACT_TASK_EXT = '.pickle'

PathFindingTask = collections.namedtuple('PathFindingTask',
                                         'title local_map start finish path'.split(' '))

CompactPathFindingTask = collections.namedtuple('CompactPathFindingTask',
                                                'map_id start finish path'.split(' '))


class TaskSet(object):
    def __init__(self, paths_dir, maps_dir):
        self.paths_dir = paths_dir
        self.map_dir = maps_dir
        self.task_names = [os.path.splitext(fn)[0] for fn in os.listdir(self.paths_dir)]
        self.maps_cache = {}

    def keys(self):
        return self.task_names

    def __getitem__(self, task_name):
        with open(os.path.join(self.paths_dir, task_name + COMPACT_TASK_EXT), 'rb') as f:
            task = cPickle.load(f)

        local_map = self.maps_cache.get(task.map_id)
        if local_map is None:
            with numpy.load(os.path.join(self.map_dir, task.map_id + COMPACT_MAP_EXT)) as f:
                local_map = f['arr_0']
            self.maps_cache[task.map_id] = local_map

        return PathFindingTask(task_name,
                               local_map,
                               task.start,
                               task.finish,
                               task.path)
