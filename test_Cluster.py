import unittest
from Cluster import Cluster, k_means
from Document import Document
from Loader import create_easy_documents, compute_homogeneity, compute_completeness
from make_dataset import create_docs

class TestCluster(unittest.TestCase):
    def test_calculate_centroid(self):
        # Create sample documents
        doc1 = Document(true_class='pos')
        doc1.tokens = {'word1': 1, 'word2': 2}
        doc2 = Document(true_class='pos')
        doc2.tokens = {'word2': 1, 'word3': 3}

        # Create cluster and calculate centroid
        cluster = Cluster(members=[doc1, doc2])
        cluster.calculate_centroid()

        # Expected centroid tokens
        expected_tokens = {'word1': 0.5, 'word2': 1.5, 'word3': 1.5}
        self.assertEqual(cluster.centroid.tokens, expected_tokens)

    def test_kmeans(self):
        # Define lexicons and length
        pos_lexicon = ['happy', 'joyful', 'pleasant']
        neg_lexicon = ['sad', 'angry', 'unpleasant']
        length = 5  # Words per document

        # Generate test documents
        pos_docs, neg_docs = create_docs(10, 10, length, pos_lexicon, neg_lexicon)

        # Create documents
        positive_docs = create_easy_documents(pos_docs, 'pos')
        negative_docs = create_easy_documents(neg_docs, 'neg')

        data = positive_docs + negative_docs

        # Run k-means clustering
        clusters = k_means(2, ['pos', 'neg'], data)

        # Check that clusters are formed
        self.assertEqual(len(clusters), 2)

    def test_compute_homogeneity(self):
        # Define lexicons and length
        pos_lexicon = ['good', 'nice', 'positive']
        neg_lexicon = ['bad', 'mean', 'negative']
        length = 5

        # Generate documents
        pos_docs, neg_docs = create_docs(3, 4, length, pos_lexicon, neg_lexicon)

        # Create documents
        positive_docs = create_easy_documents(pos_docs, 'pos')
        negative_docs = create_easy_documents(neg_docs, 'neg')

        data = positive_docs + negative_docs

        # Run k-means clustering
        clusters = k_means(2, ['pos', 'neg'], data)

        # Compute homogeneity
        homogeneity = compute_homogeneity(clusters, ['pos', 'neg'])

        # Check that homogeneity is calculated
        self.assertEqual(len(homogeneity), 2)

    def test_compute_completeness(self):
        # Define lexicons and length
        pos_lexicon = ['win', 'success', 'victory']
        neg_lexicon = ['loss', 'failure', 'defeat']
        length = 5

        # Generate documents
        pos_docs, neg_docs = create_docs(3, 1, length, pos_lexicon, neg_lexicon)

        # Create documents
        positive_docs = create_easy_documents(pos_docs, 'pos')
        negative_docs = create_easy_documents(neg_docs, 'neg')

        data = positive_docs + negative_docs

        # Run k-means clustering
        clusters = k_means(2, ['pos', 'neg'], data)

        # Compute completeness
        completeness = compute_completeness(clusters, ['pos', 'neg'])

        # Check that completeness is calculated
        self.assertEqual(len(completeness), 2)

if __name__ == '__main__':
    unittest.main()

