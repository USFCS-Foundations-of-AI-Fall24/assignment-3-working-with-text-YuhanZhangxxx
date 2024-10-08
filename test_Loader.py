import unittest
from Loader import create_easy_documents, not_stopword, convert_to_lowercase, remove_trailing_punct
from make_dataset import create_docs, generate_random_words

class TestLoader(unittest.TestCase):
    def test_workflow(self):
        # Define lexicons and length
        pos_lexicon = ['good', 'great', 'excellent']
        neg_lexicon = ['bad', 'terrible', 'awful']
        length = 10  # Words per document

        # Generate test documents
        pos_reviews, neg_reviews = create_docs(10, 10, length, pos_lexicon, neg_lexicon)

        # Create easy documents
        positive_docs = create_easy_documents(pos_reviews, 'pos',
                                              filters=[not_stopword],
                                              transforms=[convert_to_lowercase, remove_trailing_punct])
        negative_docs = create_easy_documents(neg_reviews, 'neg',
                                              filters=[not_stopword],
                                              transforms=[convert_to_lowercase, remove_trailing_punct])

        # Check document counts
        self.assertEqual(len(positive_docs), 10)
        self.assertEqual(len(negative_docs), 10)

if __name__ == '__main__':
    unittest.main()





