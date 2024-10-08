from unittest import TestCase
from Document import *


class TestDocument(TestCase):
    def test_init(self):
        d = Document(true_class='pos')
        # Since 'fish' was not added yet, the count should be 0
        self.assertEqual(d.tokens['fish'], 0)

    def test_addTokens(self):
        d = Document(true_class='pos')
        d.add_tokens(['cat', 'dog', 'fish', 'aardvark'])
        self.assertEqual(d.tokens['cat'], 1)
        self.assertEqual(d.tokens['dog'], 1)
        self.assertEqual(d.tokens['fish'], 1)
        self.assertEqual(d.tokens['aardvark'], 1)

    def test_addTokens_single_counts(self):
        d = Document(true_class='pos')
        d.add_tokens(['cat', 'dog', 'fish'])
        self.assertEqual(d.tokens['cat'], 1)
        self.assertEqual(d.tokens['dog'], 1)
        self.assertEqual(d.tokens['fish'], 1)
        self.assertEqual(d.tokens['aardvark'], 0)


class TestDistance(TestCase):
    def test_euclidean_distance(self):
        d1 = Document(true_class='pos')
        d1.add_tokens(['cat', 'dog', 'fish'])
        d2 = Document(true_class='pos')
        d2.add_tokens(['cat', 'dog', 'fish'])
        self.assertEqual(euclidean_distance(d1, d2), 0)
        d3 = Document(true_class='pos')
        d3.add_tokens(['cat', 'bunny', 'fish'])
        self.assertEqual(euclidean_distance(d1, d3), 2)

    def test_cosine_similarity(self):
        d1 = Document(true_class='pos')
        d1.add_tokens(['cat', 'dog', 'fish'])
        d2 = Document(true_class='pos')
        d2.add_tokens(['cat', 'dog', 'fish'])
        self.assertAlmostEqual(cosine_similarity(d1, d2), 1.0, places=4)
        d3 = Document(true_class='pos')
        d3.add_tokens(['cat', 'bunny', 'fish'])
        self.assertAlmostEqual(cosine_similarity(d1, d3), 0.6667, places=4)
