#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Testcase of da
# This code is available under the MIT License.
# (c)2012 Nakatani Shuyo / Cybozu Labs Inc.
# (c) 2017 Demonchy Clement


import unittest
from ldig import da

class TestDoubleArray(unittest.TestCase):
    def test1(self):
        trie = da.DoubleArray(verbose=False)
        trie.initialize(["cat"])
        self.assertEqual(trie.N, 4)
        self.assertTrue(trie.get("ca") is None)
        self.assertTrue(trie.get("xxx") is None)
        self.assertEqual(trie.get("cat"), 0)

    def test2(self):
        trie = da.DoubleArray()
        trie.initialize(["cat", "dog"])
        self.assertEqual(trie.N, 7)
        self.assertTrue(trie.get("ca") is None)
        self.assertTrue(trie.get("xxx") is None)
        self.assertEqual(trie.get("cat"), 0)
        self.assertEqual(trie.get("dog"), 1)

    def test3(self):
        trie = da.DoubleArray(verbose=False)
        trie.initialize(["ca", "cat", "deer", "dog", "fox", "rat"])
        self.assertEqual(trie.N, 17)
        self.assertTrue(trie.get("c") is None)
        self.assertEqual(trie.get("ca"), 0)
        self.assertEqual(trie.get("cat"), 1)
        self.assertEqual(trie.get("deer"), 2)
        self.assertEqual(trie.get("dog"), 3)
        self.assertTrue(trie.get("xxx") is None)

    def test4(self):
        trie = da.DoubleArray()
        self.assertRaises(Exception, trie.initialize, ["cat", "ant"])

    def test5(self):
        trie = da.DoubleArray(verbose=False)
        trie.initialize(["ca", "cat", "deer", "dog", "fox", "rat"])

        r = trie.extract_features("")
        self.assertEqual(len(r), 0)

        r = trie.extract_features("cat")
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0], 1)
        self.assertEqual(r[1], 1)

        r = trie.extract_features("deerat")
        self.assertEqual(len(r), 2)
        self.assertEqual(r[2], 1)
        self.assertEqual(r[5], 1)

if __name__ == "__main__":
    unittest.main()

