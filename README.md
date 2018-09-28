ldig (Language Detection with Infinity Gram)
======================

Updated for Python3

This is a prototype of language detection for short message service (twitter).
with 99.1% accuracy for 17 languages


Usage
------

  1. Install with setup.py install
  2. import ldig from ldig
  3. lmodel = ldig()
  4. ldig.detect_text(some text)
  
  
Data format
------

As input data, Each tweet is one line in text file as the below format.

    [label]\t[some metadata separated '\t']\t[text without '\t']

[label] is a language name alike en, de, fr and so on.
It is also optional as metadata.
(ldig doesn't use metadata and label for detection, of course :D)

The output data of lidg is as the below.

    [correct label]\t[detected label]\t[original metadata and text]


Estimation Tool
----

ldig has a estimation tool.

    ./server.py -m [model directory]

Open http://localhost:48000 and input target text into textarea.
Then ldig outputs language probabilities and feature parameters in the text.


Supported Languages
------

- cs	Czech
- da	Dannish
- de	German
- en	English
- es	Spanish
- fi	Finnish
- fr	French
- id	Indonesian
- it	Italian
- nl	Dutch
- no	Norwegian
- pl	Polish
- pt	Portuguese
- ro	Romanian
- sv	Swedish
- tr	Turkish
- vi	Vietnamese


Documents
------

- [Presentation in English](http://www.slideshare.net/shuyo/short-text-language-detection-with-infinitygram-12949447)
- [Presentation in Japanese](http://www.slideshare.net/shuyo/gram-10286133)

- Blog Articles about ldig
  - [Language Detection for twitter with 99.1% Accuracy](http://shuyo.wordpress.com/2012/02/21/language-detection-for-twitter-with-99-1-accuracy/)
  - [Precision and Recall of ldig (twitter language detection)](http://shuyo.wordpress.com/2012/03/02/precision-and-recall-of-ldig-twitter-language-detection/)
  - [Estimation of ldig (twitter Language Detection) for LIGA dataset](http://shuyo.wordpress.com/2012/03/02/estimation-of-ldig-twitter-language-detection-for-liga-dataset/)
  - [Why is Norwegian and Danish identification difficult?](http://shuyo.wordpress.com/2012/03/07/why-is-norwegian-and-danish-identification-difficult/)


Copyright & License
-----
- (c)2011-2012 Nakatani Shuyo / Cybozu Labs Inc. All rights reserved.
- (c) 2017 Demonchy Clement. All rights reserved
- All codes and resources are available under the MIT License.

