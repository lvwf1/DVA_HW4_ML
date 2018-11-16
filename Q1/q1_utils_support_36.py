# uncompyle6 version 3.2.4
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.7.0 (default, Jun 28 2018, 13:15:42) 
# [GCC 7.2.0]
# Embedded file name: /Users/nilakshdas/Desktop/hw4-skeleton/Q1/q1_utils_support_36.py
# Compiled at: 2018-10-31 02:54:08
# Size of source mod 2**32: 5095 bytes
from __future__ import print_function, absolute_import
import re, mmap
from struct import pack
import os, sys, json, argparse
from warmup import get_memory_map_from_binary_file, parse_memory_map
from pagerank import pagerank

class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='#', empty_symbol='.', output=sys.stderr):
        if not len(symbol) == 1:
            raise AssertionError
        self.total = total
        self.width = width
        self.symbol = symbol
        self.empty_symbol = empty_symbol
        self.output = output
        self.fmt = re.sub('(?P<name>%\\(.+?\\))d', '\\g<name>%dd' % len(str(total)), fmt)
        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + self.empty_symbol * (self.width - size) + ']'
        args = {'total':self.total,  'bar':bar, 
         'current':self.current, 
         'percent':percent * 100, 
         'remaining':remaining}
        print('\r' + self.fmt % args + ' (%d/%d)' % (self.current, self.total), file=self.output, end='')

    def update(self, progress):
        self.current = progress
        self()

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)


def skip_header_read(f):
    line_number = 0
    for line in f:
        line_number += 1
        if not line[0] == '#':
            if line[0] == '\n':
                pass
            else:
                yield (
                 line, line_number)


def file_len(file_path):
    """ Count number of lines in a file."""
    f = open(file_path)
    lines = 0
    buf_size = 1048576
    read_f = f.read
    buf = read_f(buf_size)
    while buf:
        lines += buf.count('\n')
        buf = read_f(buf_size)

    return lines


def test_warmup_submission(filepath):
    try:
        expected = [(i, i ** 2) for i in range(1, 42, 2)]
        for i, item in enumerate(parse_memory_map(get_memory_map_from_binary_file(filepath)[1])):
            if not item == expected[i]:
                raise AssertionError('Output is incorrect')

        print(True)
    except Exception as ex:
        print(False)
        print(ex)


def convert(edge_list):
    output_file_prefix = ('').join(edge_list.split('.')[:-1])
    edge_path = '%s.bin' % output_file_prefix
    index_path = '%s.idx' % output_file_prefix
    meta_path = '%s.json' % output_file_prefix
    max_node = 0
    node_count = 0
    print('Counting lines...', file=sys.stderr)
    num_lines = file_len(edge_list)
    print('Converting...', file=sys.stderr)
    pb = ProgressBar(num_lines)
    pb()
    with open(edge_list) as (edge_list_file):
        with open(edge_path, mode='wb') as (edge_file):
            with open(index_path, mode='wb') as (index_file):
                cur = 0
                cur_len = 0
                last_source = None
                last_write = 0
                for line, line_number in skip_header_read(edge_list_file):
                    if line_number % 100000 == 0:
                        pb.update(line_number)
                    source, target = [int(x) for x in line.split()]
                    edge_file.write(pack('<ii', source, target))
                    max_node = max(max_node, max(source, target))
                    if last_source != source:
                        if last_source is not None:
                            if last_source != last_write + 1:
                                for _ in range(last_source - last_write - 1):
                                    index_file.write(pack('<qq', -1, -1))

                            last_write = last_source
                            index_file.write(pack('<qq', cur - cur_len, cur_len))
                        cur_len = 1
                        node_count += 1
                    else:
                        cur_len += 1
                    cur += 1
                    last_source = source

                if last_source != last_write + 1:
                    for i in range(last_write + 1, last_source):
                        index_file.write(pack('<qq', -1, -1))
                        last_write = i

                index_file.write(pack('<qq', cur - cur_len, cur_len))
                for i in range(last_write + 1, max_node + 1):
                    index_file.write(pack('<qq', -1, -1))

                for _ in range(15):
                    index_file.write(pack('<q', -1))
                    edge_file.write(pack('<i', -1))

    pb.done()
    with open(meta_path, 'w') as (meta_file):
        meta_file.write(json.dumps({'edge_path':os.path.basename(edge_path),  'index_path':os.path.basename(index_path), 
         'edge_count':cur, 
         'node_count':node_count, 
         'max_node':max_node},
          indent=2,
          sort_keys=True))
    print('Meta data wrote to: %s.' % meta_path, file=sys.stderr)
# okay decompiling q1_utils_support_36.pyc
