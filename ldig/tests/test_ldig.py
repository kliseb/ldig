#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Testcase of ldig
# This code is available under the MIT License.
# (c)2012 Nakatani Shuyo / Cybozu Labs Inc.
# (c) 2017 Demonchy Clement

import unittest
from ldig import ldig
import os

PATH_SCRIPT = os.path.dirname(os.path.realpath(__file__))
class TestLdig(unittest.TestCase):
    """Normalization test"""
    def assertNormalize(self, org, norm):
        self.assertEqual(ldig.normalize_text(org), ("", norm, org))

    def testNormalizeRT(self):
        self.assertNormalize("RT RT RT RT RT I'm a Superwoman", "I'm a superwoman")

    def testNormalizeLaugh(self):
        self.assertNormalize("ahahahah", "ahahah")
        self.assertNormalize("hahha", "haha")
        self.assertNormalize("hahaa", "haha")
        self.assertNormalize("ahahahahhahahhahahaaaa", "ahaha")

    def testLowerCaseWithTurkish(self):
        self.assertNormalize("I", "I")
        self.assertNormalize("İ", "i")
        self.assertNormalize("i", "i")
        self.assertNormalize("ı", "ı")

        self.assertNormalize("Iİ", "Ii")
        self.assertNormalize("Iı", "Iı")

    def testDetectLangText(self):
        ldig_object = ldig.ldig()
        self.assertEqual(ldig_object.detect_text("Coucou")[1],"fr")
        self.assertEqual(ldig_object.detect_text("CABOSUN Trial of Upfront Cabozantinib in Metastatic RCC http://ht.ly/P4tr30b4tTr")[1],"en")
        self.assertEqual(ldig_object.detect_text("@dr_l_alexandre C est de l acromegalie, non ?")[1],"fr")
        # text too small, so url need to be remove else it will fail
        self.assertEqual(ldig_object.detect_text(" acromégalie. https://t.co/JncayILmyG")[1],"fr")

    def testMultipleInit(self):
        #try to detect crash that happen randomly when initialising ldig and more precisly load_da
        for i in range(20):
            ldig.ldig()
            
if __name__ == '__main__':
    unittest.main()
