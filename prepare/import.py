#!/usr/bin/env python

import argparse
import cPickle
import collections
import hashlib
import os

import lxml
import lxml.etree
import numpy
from pathenv.tasks import PathFindingTask, CompactPathFindingTask


def load_from_xml(fname, ctor=PathFindingTask):
    try:

        with open(fname, 'r') as f:
            tree = lxml.etree.parse(f)

        title = os.path.splitext(os.path.basename(fname))[0]

        local_map = numpy.array([map(int, row.split(' '))
                                 for row in tree.xpath('/root/map/grid/row/text()')],
                                dtype='uint8')
        start_x = int(tree.xpath('/root/map/startx/text()')[0])
        start_y = int(tree.xpath('/root/map/starty/text()')[0])
        finish_x = int(tree.xpath('/root/map/finishx/text()')[0])
        finish_y = int(tree.xpath('/root/map/finishy/text()')[0])

        sections = tree.xpath('/root/log[1]/hplevel/section')
        sections.sort(key=lambda n: int(n.get('number')))
        path = [(int(s.get('start.y')),
                 int(s.get('start.x')))
                for s in sections]
        if path:
            path.append((int(sections[-1].get('finish.y')),
                         int(sections[-1].get('finish.x'))))
        result = ctor(title,
                      local_map,
                      (start_y, start_x),
                      (finish_y, finish_x),
                      path)
        return result
    except:
        return None


def save_to_compact(task, maps_dir, paths_dir):
    map_id = hashlib.md5(task.local_map.tostring()).hexdigest()
    map_fname = os.path.join(maps_dir, map_id)
    if not os.path.exists(map_fname + '.npz'):
        numpy.savez_compressed(map_fname, task.local_map)

    compact = CompactPathFindingTask(map_id, task.start, task.finish, task.path)
    task_fname = os.path.join(paths_dir, task.title + '.pickle')
    with open(task_fname, 'wb') as f:
        cPickle.dump(compact, f, 2)


if __name__ == "__main__":
    aparser = argparse.ArgumentParser()
    aparser.add_argument('src_dir', type=str)
    aparser.add_argument('target_dir', type=str)

    args = aparser.parse_args()
    maps_subdir = 'maps'
    paths_subdir = 'paths'

    maps_dir = os.path.join(args.target_dir, maps_subdir)
    paths_dir = os.path.join(args.target_dir, paths_subdir)

    for in_fname in os.listdir(args.src_dir):
        task = load_from_xml(os.path.join(args.src_dir, in_fname))
        if task is None:
            continue
        save_to_compact(task, maps_dir, paths_dir)
