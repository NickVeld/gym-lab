#!/usr/bin/env python

import argparse
import cPickle
import collections
import hashlib
import os
import random

import numpy
from scipy.spatial.distance import euclidean

from pathenv.tasks import BY_PIXEL_ACTION_DIFFS, CompactPathFindingTask, PathFindingTask

StepResult = collections.namedtuple('StepResult',
                                    'must_continue best_next new_variants_with_ratings'.split(' '))


class BaseSearchAlgo(object):
    def __init__(self):
        pass

    def reset(self, local_map, start, finish):
        self.local_map = local_map
        self.start = start
        self.finish = finish

        self.queue = [self.start]
        self.ratings = {self.start: 0}
        self.backrefs = {self.start: self.start}
        self.visited_nodes = set()

    def walk_to_finish(self):
        while self.step().must_continue:
            pass

    def step(self):
        if self.goal_achieved():
            return StepResult(False, self.finish, [])

        while len(self.queue) > 0:
            self._reorder_queue()
            best_next = self.queue.pop()
            self.visited_nodes.add(best_next)

            if self.goal_achieved():
                return StepResult(False, self.finish, [])

            new_variants_with_ratings = self._gen_new_variants(best_next)
            if len(new_variants_with_ratings) == 0:
                continue

            self.queue.extend(p for p, _ in new_variants_with_ratings)
            self.ratings.update(new_variants_with_ratings)
            self.backrefs.update((new_point, best_next) for new_point, _ in new_variants_with_ratings)
            return StepResult(True, best_next, new_variants_with_ratings)

        return StepResult(False, None, [])

    def update_ratings(self, updates):
        self.ratings.update(updates)

    def goal_achieved(self):
        return self.finish in self.visited_nodes

    def get_best_path(self):
        self.walk_to_finish()
        if not self.goal_achieved():
            return None

        result = [self.finish]
        while result[-1] != self.start:
            result.append(self.backrefs[result[-1]])
        result.reverse()
        return result

    def _reorder_queue(self):
        self.queue.sort(key=self.ratings.__getitem__)

    def _gen_new_variants(self, pos):
        y, x = pos
        all_new_points = ((y + dy, x + dx)
                          for dy, dx
                          in BY_PIXEL_ACTION_DIFFS.viewvalues())
        return [(point, -euclidean(point, self.finish))
                for point in all_new_points
                if (not point in self.backrefs)
                and (0 <= point[0] < self.local_map.shape[0])
                and (0 <= point[1] < self.local_map.shape[1])
                and (self.local_map[point] == 0)]


def _gen_point(map_shape):
    return (random.randint(0, map_shape[0] - 1),  # randint is inclusive
            random.randint(0, map_shape[1] - 1))


def save_to_compact(task, maps_dir, paths_dir):
    map_id = hashlib.md5(local_map.tostring()).hexdigest()
    map_fname = os.path.join(maps_dir, map_id)
    if not os.path.exists(map_fname + '.npz'):
        numpy.savez_compressed(map_fname, task.local_map)

    compact = CompactPathFindingTask(map_id, task.start, task.finish, task.path)
    task_fname = os.path.join(paths_dir, task.title + '.pickle')

    with open(task_fname, 'wb') as f:
        cPickle.dump(compact, f, 2)


if __name__ == '__main__':
    aparser = argparse.ArgumentParser()
    aparser.add_argument('-n',
                         type=int,
                         default=10,
                         help='How much tasks to generate')
    aparser.add_argument('map_fname',
                         type=str,
                         help='Path to .npz-file with map to generate tasks for')
    aparser.add_argument('out_dir',
                         type=str,
                         help='Where to put tasks')

    args = aparser.parse_args()

    path_builder = BaseSearchAlgo()

    map_dir = os.path.dirname(args.map_fname)
    with numpy.load(args.map_fname) as f:
        local_map = f['arr_0']
    for i in xrange(args.n):
        while True:
            start = _gen_point(local_map.shape)
            finish = _gen_point(local_map.shape)
            if local_map[start] != 0 or local_map[finish] != 0:
                continue

            path_builder.reset(local_map, start, finish)
            path = path_builder.get_best_path()
            if not path is None:
                break
        task = PathFindingTask(str(i), local_map, start, finish, path)
        print task
        save_to_compact(task, map_dir, args.out_dir)
