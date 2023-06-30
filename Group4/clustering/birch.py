import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from numpy.typing import NDArray


class Birch:
    """
    A simplified approach to the Birch algorithm


    ## Params
         branching_factor, int, default=40 :
            Number of Cluster Features a Node can hold
         threshold, float, default=0.8:
            The radius. If updating the Subcluster with the new point would exceed this radius, the new point will create a new Subcluster
         n_cluster, int, default=None:
            If a number is given, a global AgglomerativeClustering will combine
            subcluster centroids until n_clusters is reached
        leaf_factor, int, default=None:
            The branching factor for leaf nodes
            Default = 2*branching_factor
        predict, bool, default=True:
            If True, the Birch object will predict the labels of the given data X
            If you only want to build the tree, set predict=False.
            You wont be able to acces the labels afterwards

    ## How to use
        create Birch object -> Use fit() function to build the Tree

        Afterwards you can access following variables:
            all_centroids (n_centroids, n_features) : generated centroids of the subclusters
            labels (n_samples,) : predicted labels regarding the given points X: 1-d numpy array

    ## References:
        [1] Tian Zhang, Raghu Ramakrishnan, and Miron Livny. 1996.
            BIRCH: an efficient data clustering method for very large databases.
            SIGMOD Rec. 25, 2 (June 1996), 103-114. https://doi.org/10.1145/235968.233324

    """

    def __init__(
        self,
        branching_factor: int = 40,
        threshold: float = 0.8,
        n_cluster: int = None,
        leaf_factor: int = None,
        predict=True,
    ) -> None:
        self.bf = branching_factor
        self.threshold = threshold
        self.root: Node
        self.n_cluster = n_cluster
        self.predict_value = predict
        if leaf_factor is None:
            self.leaf_factor = self.bf * 2
        else:
            self.leaf_factor = leaf_factor

    def fit(self, X: NDArray) -> None:
        """
        Build the CF Tree

        ## Params
            X: 2-dimensional array (n_samples, n_features)
        """
        assert len(X.shape) == 2, print("X must be 2-dimensional in form of (n_samples,n_features)")
        assert self.bf > 2 and self.leaf_factor > 2, print("branching and leaf factor must be greater than 2")

        X = np.array(X).copy()

        _, n_features = X.shape

        first_node = Node(
            bf=self.bf, threshold=self.threshold, isLeaf=True, n_features=n_features, leaf_factor=self.leaf_factor
        )
        self.root = first_node

        for point in X:
            cf = CF(LS=point)
            split_status = self.root.input_cf(cf=cf)

            if split_status:
                sc1, sc2, n1, n2 = self.root.split_node(self.root, self.bf, self.threshold, self.leaf_factor)

                self.root = Node(
                    bf=self.bf,
                    threshold=self.threshold,
                    isLeaf=False,
                    n_features=n_features,
                    leaf_factor=self.leaf_factor,
                )
                self.root.append_cf(sc1, n1)
                self.root.append_cf(sc2, n2)

        self.all_centroids = np.concatenate(self.get_subcluster())
        if self.n_cluster is not None:
            assert len(self.all_centroids) >= self.n_cluster, print(
                f"n_clusters({self.n_cluster}) must be higher than fitted centroids ({self.all_centroids})"
            )
            self.all_centroids = self.global_clutering()
        if self.predict_value:
            self.labels = self.predict(X)

    def global_clutering(self) -> NDArray:
        """
        Internal.\n
        Optional step, that combines the center_points to the given number of n_cluster
        ## Returns
            ndarray (n_cluster,c_feature) of new center points
        """
        global_cluster = AgglomerativeClustering(n_clusters=self.n_cluster)
        new_labels = global_cluster.fit_predict(self.all_centroids)
        new_centers = []
        for label in np.unique(new_labels):
            current = self.all_centroids[new_labels == label]
            new_center = current.mean(axis=0)
            new_centers.append(new_center)

        return np.array(new_centers)

    def predict(self, X: NDArray) -> NDArray:
        """
        Predict based on fitted data
        ## Params
            X: 2-dimensional array (n_samples, n_features)

        ## Returns
            new labels : (n_samples,)
        """
        dist = cdist(X, self.all_centroids)
        labels = np.argmin(dist, axis=1)
        return labels

    def get_subcluster(self) -> NDArray:
        """
        Internal.\n
        Gets all Subcluster Information: So all CFs from Leaf Nodes
        ## Returns
            Array of leafs
        """
        all_leafs = []
        random_leaf = self.root
        while not random_leaf.isLeaf:
            random_leaf = random_leaf.CFs[0][1]

        while random_leaf.prev_leaf is not None:
            random_leaf = random_leaf.prev_leaf

        all_leafs.append(random_leaf.centers[: len(random_leaf.CFs)])
        while random_leaf.next_leaf is not None:
            random_leaf = random_leaf.next_leaf
            all_leafs.append(random_leaf.centers[: len(random_leaf.CFs)])

        return all_leafs


class Node:
    """
    Internal class.

    Nodes of the tree containing all informations

    ## Params:
        bf, int: branching_factor
        threshold, float: the maximum radius of a cluster
        isLeaf, bool: If Node is a leaf node or not
        n_features, int: Number of features of the data, used to initiale centers with 0s
        leaf_factor, int: branching_factor of leaf nodes
    """

    def __init__(self, bf: int, threshold: float, isLeaf: bool, n_features: int, leaf_factor: int) -> None:
        self.CFs = []
        self.bf = bf
        self.isLeaf = isLeaf
        self.threshold = threshold
        self.n_features = n_features
        self.prev_leaf = None
        self.next_leaf = None
        self.leaf_factor = leaf_factor
        if self.leaf_factor > self.bf:
            self.centers = np.zeros((self.leaf_factor + 1, n_features))
        else:
            self.centers = np.zeros((bf + 1, n_features))

    def input_cf(self, cf: object) -> bool:
        """
        Trys to fit a new point into a Node

        This is a recursive function

        return True : Node needs to be split, split will be handled by parent node
        return False: Split is not necessary
        """

        # First Node
        if not self.CFs:
            self.append_cf(cf)
            return False

        dist = np.linalg.norm(self.centers[: len(self.CFs)] - cf.center, axis=1)
        idx = np.argmin(dist)
        target_cf = self.CFs[idx][0]

        if self.isLeaf:
            radius = ((target_cf.SS + cf.SS) - np.sum(((target_cf.LS + cf.LS) ** 2) / (target_cf.N + cf.N))) / (
                target_cf.N + cf.N
            )

            if radius <= self.threshold**2:
                # all good, dont forget to update node
                target_cf.update(cf=cf)
                self.centers[idx] = target_cf.center
                return False

            else:
                # out of range
                # Even if not enough space, still add Subcluster for easier updating
                self.append_cf(cf)

                if len(self.CFs) <= self.leaf_factor:
                    return False  # New cluster can be added to Node
                else:
                    return True  # New cluster wont have enough space

        if not self.isLeaf:
            # Create recursive structure to traverse tree
            split_status = self.CFs[idx][1].input_cf(cf)

            if split_status:
                # Node needs to be split
                sc1, sc2, n1, n2 = self.split_node(self.CFs[idx][1], self.bf, self.threshold, self.leaf_factor)

                # Overwrite the cf and node of the splittet node and append the other node
                self.CFs[idx] = (sc1, n1)
                self.centers[idx] = sc1.center
                self.append_cf(sc2, n2)

                if self.bf < len(self.CFs):
                    return True
                return False

            else:
                # Easy, just update according cf
                target_cf.update(cf=cf)
                self.centers[idx] = self.CFs[idx][0].center
                return False

    def append_cf(self, cf: object, child_node=None) -> None:
        """Append new cf to Node"""
        self.CFs.append((cf, child_node))
        self.centers[len(self.CFs) - 1] = cf.center

    def split_node(self, node: object, bf: int, threshold: int, leaf_factor: int) -> object:
        """
        If insertion of CF exceeds branching_factor the Node needs to be split

        Remarks:
            The furthest CFs will build the starting point of node1 and node2
            The old CF will be dereferenced and 2 new will be added
        """
        new_cf1 = CF()
        new_cf2 = CF()
        node1 = Node(
            bf=bf, threshold=threshold, isLeaf=node.isLeaf, n_features=node.n_features, leaf_factor=leaf_factor
        )
        node2 = Node(
            bf=bf, threshold=threshold, isLeaf=node.isLeaf, n_features=node.n_features, leaf_factor=leaf_factor
        )

        distances = cdist(node.centers[: len(node.CFs)], node.centers[: len(node.CFs)], "euclidean")
        n1, n2 = np.unravel_index(distances.argmax(), distances.shape)
        f1 = node.centers[n1]
        f2 = node.centers[n2]

        for cf in node.CFs:
            if np.linalg.norm(cf[0].center - f1) < np.linalg.norm(cf[0].center - f2):
                node1.append_cf(cf[0], cf[1])
                new_cf1.update(cf[0])
            else:
                node2.append_cf(cf[0], cf[1])
                new_cf2.update(cf[0])

        if node.isLeaf:
            # to get all cluster in fast manner, leaf nodes get a pointer to next and previos leaf
            # In summary this puts node1 and node2 in the pointer-chain and excludes the given node
            node1.prev_leaf = node.prev_leaf
            node1.next_leaf = node2
            node2.prev_leaf = node1
            node2.next_leaf = node.next_leaf
            if node.prev_leaf is not None:
                node.prev_leaf.next_leaf = node1
            if node.next_leaf is not None:
                node.next_leaf.prev_leaf = node2
        return new_cf1, new_cf2, node1, node2


class CF:
    """
    Helper class
    Defines properties of cluster feature (CF):
        N = number of points in CF
        LS = Linear sum
        SS = Linear sum
        center = not required, but appended for easier calculation

    ## Params:
        LS, default=None:
            Only LS is needed for init, as rest is logical from there
    """

    def __init__(self, LS=None) -> None:
        if LS is None:
            self.N = 0
            self.LS = 0
            self.center = 0
            self.SS = 0

        else:
            self.N = 1
            self.LS = LS
            self.center = LS
            self.SS = np.dot(LS, LS)

    def update(self, cf: object) -> None:
        self.N += cf.N
        self.LS += cf.LS
        self.SS += cf.SS
        self.center = self.LS / self.N
