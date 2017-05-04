#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import collections
import numpy
import logging
import time

logger = logging.getLogger(__name__)

# Double Array for static ordered data
# This code is available under the MIT License.
# (c)2011 Nakatani Shuyo / Cybozu Labs Inc.

class DoubleArray(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def validate_list(self, sorted_list):
        pre = ""
        for i, line in enumerate(sorted_list):
            if pre >= line:
                raise Exception("list has not ascent order at %d" % (i+1))
            pre = line

    def initialize(self, list_to_initialize):
        self.validate_list(list_to_initialize)

        self.N = 1
        self.base  = [-1]
        self.check = [-1]
        self.value = [-1]

        max_index = 0
        queue = collections.deque([(0, 0, len(list_to_initialize), 0)])
        while len(queue) > 0:
            index, left, right, depth = queue.popleft()
            if depth >= len(list_to_initialize[left]):
                self.value[index] = left
                left += 1
                if left >= right: continue

            # get branches of current node
            stack = collections.deque([(right, -1)])
            cur, c1 = (left, ord(list_to_initialize[left][depth]))
            result = []
            while len(stack) >= 1:
                while c1 == stack[-1][1]:
                    cur, c1 = stack.pop()
                mid = (cur + stack[-1][0]) / 2
                if cur == mid:
                    result.append((cur + 1, c1))
                    cur, c1 = stack.pop()
                else:
                    c2 = ord(list_to_initialize[mid][depth])
                    if c1 != c2:
                        stack.append((mid, c2))
                    else:
                        cur = mid

            # search empty index for current node
            v0 = result[0][1]
            j = - self.check[0] - v0
            while any(j + v < self.N and self.check[j + v] >= 0 for right, v in result):
                j = - self.check[j + v0] - v0
            tail_index = j + result[-1][1]
            if max_index < tail_index:
                max_index = tail_index
                self.extend_array(tail_index + 2)

            # insert current node into DA
            self.base[index] = j
            depth += 1
            for right, v in result:
                child = j + v
                self.check[self.base[child]] = self.check[child]
                self.base[-self.check[child]] = self.base[child]
                self.check[child] = index
                queue.append((child, left, right, depth))
                left = right

        self.shrink_array(max_index)

    def extend_array(self, max_cand):
        if self.N < max_cand:
            new_N = 2 ** int(numpy.ceil(numpy.log2(max_cand)))
            self.log("extend DA : %d => (%d) => %d", (self.N, max_cand, new_N))
            self.base.extend(    n - 1 for n in xrange(self.N, new_N))
            self.check.extend( - n - 1 for n in xrange(self.N, new_N))
            self.value.extend(     - 1 for n in xrange(self.N, new_N))
            self.N = new_N

    def shrink_array(self, max_index):
        self.log("shrink DA : %d => %d", (self.N, max_index + 1))
        self.N = max_index + 1
        self.check = numpy.array(self.check[:self.N])
        self.base = numpy.array(self.base[:self.N])
        self.value = numpy.array(self.value[:self.N])

        not_used = self.check < 0
        self.check[not_used] = -1
        not_used[0] = False
        self.base[not_used] = self.N

    def log(self, format_log, param):
        if self.verbose:
            logger.info("-- %s, %s" % (time.strftime("%Y/%m/%d %H:%M:%S"), format_log % param))

    def save(self, filename):
        numpy.savez(filename, base=self.base, check=self.check, value=self.value)

    def load(self, filename):
        with numpy.load(filename) as loaded:
            self.base = loaded['base']
            self.check = loaded['check']
            self.value = loaded['value']
            self.N = self.base.size

    def add_element(self, s, v):
        pass

    def get_subtree(self, s):
        cur = 0
        for c in iter(s):
            v = ord(c)
            next_element = self.base[cur] + v
            if next_element >= self.N or self.check[next_element] != cur:
                return None
            cur = next_element
        return cur

    def get_child(self, c, subtree):
        v = ord(c)
        next_element = self.base[subtree] + v
        if next >= self.N or self.check[next_element] != subtree:
            return None
        return next_element

    def get(self, s):
        cur = self.get_subtree(s)
        if cur >= 0:
            value = self.value[cur]
            if value >= 0: return value
        return None

    def get_value(self, subtree):
        return self.value[subtree]

    def extract_features(self, st):
        events = dict()
        l = len(st)
        clist = [ord(c) for c in iter(st)]
        N = self.N
        base = self.base
        check = self.check
        value = self.value
        for i in xrange(l):
            pointer = 0
            for j in xrange(i, l):
                next_element = base[pointer] + clist[j]
                if next_element >= N or check[next_element] != pointer: break
                id_value = value[next_element]
                if id_value >= 0:
                    events[id_value] = events.get(id_value, 0) + 1
                pointer = next_element
        return events

