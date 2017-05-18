#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Testcase of ldig
# This code is available under the MIT License.
# (c)2012 Nakatani Shuyo / Cybozu Labs Inc.
# (c) 2017 Demonchy Clement
from __future__ import absolute_import, unicode_literals
import unittest
from ldig import ldig
import os

PATH_SCRIPT = os.path.dirname(os.path.realpath(__file__))
class TestLdig(unittest.TestCase):
    """Normalization test"""
    def assertNormalize(self, org, norm):
        self.assertEqual(ldig.normalize_text(org), ("", norm, org))

    def testNormalizeRT(self):
        self.assertNormalize(u"RT RT RT RT RT I'm a Superwoman", u"I'm a superwoman")

    def testNormalizeLaugh(self):
        self.assertNormalize(u"ahahahah", u"ahahah")
        self.assertNormalize(u"hahha", u"haha")
        self.assertNormalize(u"hahaa", u"haha")
        self.assertNormalize(u"ahahahahhahahhahahaaaa", u"ahaha")

    def testLowerCaseWithTurkish(self):
        self.assertNormalize(u"I", u"I")
        self.assertNormalize(u"İ", u"i")
        self.assertNormalize(u"i", u"i")
        self.assertNormalize(u"ı", u"ı")

        self.assertNormalize(u"Iİ", u"Ii")
        self.assertNormalize(u"Iı", u"Iı")

    def testDetectLangText(self):
        ldig_object = ldig.ldig()
        self.assertEqual(ldig_object.detect_text("Coucou")[1],"fr")
        self.assertEqual(ldig_object.detect_text("CABOSUN Trial of Upfront Cabozantinib in Metastatic RCC http://ht.ly/P4tr30b4tTr")[1],"en")
        self.assertEqual(ldig_object.detect_text("@dr_l_alexandre C est de l acromegalie, non ?")[1],"fr")

    def testDetectLangInFile(self):
        ldig_object = ldig.ldig()
        prob_and_lang = ldig_object.detect_file([os.path.join(PATH_SCRIPT ,"data/tweets.txt")])
        lang = [p_a_l[1] for p_a_l in prob_and_lang]
        self.assertListEqual(lang,["fr","tr","fr","en","en","en","en","en","en","fr","en","en","fr","fr","fr","fr","fr","fr","sv","fr"])
if __name__ == '__main__':
    unittest.main()
