import random
from Document import Document, cosine_similarity
from collections import defaultdict

class Cluster:
    # Group of documents
    def __init__(self, centroid=None, members=None):
        self.centroid = centroid if centroid else Document(true_class='pos')
        self.members = members if members else []

    def __repr__(self):
        return f"{self.centroid} {len(self.members)}"

    def calculate_centroid(self):
        if not self.members:
            self.centroid = Document(true_class=self.centroid.true_class)
            return

        summed_tokens = defaultdict(float)
        for doc in self.members:
            for token, count in doc.tokens.items():
                summed_tokens[token] += count

        num_members = len(self.members)
        averaged_tokens = {token: count / num_members for token, count in summed_tokens.items()}

        centroid_doc = Document(true_class=self.centroid.true_class)
        for token, count in averaged_tokens.items():
            centroid_doc.tokens[token] = count

        self.centroid = centroid_doc

def k_means(n_clusters, true_classes, data, max_iterations=100):
    clusters = []
    for i in range(n_clusters):
        random_doc = random.choice(data)
        clusters.append(Cluster(centroid=random_doc))

    for iteration in range(max_iterations):
        for cluster in clusters:
            cluster.members = []

        for doc in data:
            similarities = [cosine_similarity(doc, cluster.centroid) for cluster in clusters]
            best_cluster_index = similarities.index(max(similarities))
            clusters[best_cluster_index].members.append(doc)

        old_centroids = [cluster.centroid.tokens.copy() for cluster in clusters]

        for cluster in clusters:
            cluster.calculate_centroid()

        converged = True
        for idx, cluster in enumerate(clusters):
            if cluster.centroid.tokens != old_centroids[idx]:
                converged = False
                break

        if converged:
            break

    return clusters






