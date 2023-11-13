from typing import List
import numpy as np

from PointSet import PointSet, FeaturesTypes


class Tree:
    """A decision Tree

    Attributes
    ----------
        points : PointSet
            The training points of the tree
    """

    def __init__(self,
                 features: List[List[float]],
                 labels: List[bool],
                 types: List[FeaturesTypes],
                 h: int = 1,
                 min_split_points: int = 1):
        """
        Parameters
        ----------
            labels : List[bool]
                The labels of the training points.
            features : List[List[float]]
                The features of the training points. Each sublist
                represents a single point. The sublists should have
                the same length as the `types` parameter, while the
                list itself should have the same length as the
                `labels` parameter.
            types : List[FeaturesTypes]
                The types of the features.
        """

        self.points = PointSet(features, labels, types)

        self.left_child = None
        self.right_child = None
        self.h = h
        self.min_split_points = min_split_points

        # Stop the splitting if the tree is alredy created or the points are alredy classified
        if h <= 0 or len(self.points.features) < self.min_split_points or len(np.unique(self.points.labels)) <= 1:
            self.left_child = None
            self.right_child = None
            return

        best_feature_id, _ = self.points.get_best_gain(self.min_split_points)

        value = self.points.get_best_threshold()
        if value is None:
            value = 0

        # Stop the splitting if best feature is not found
        if best_feature_id == -1:
            self.left_child = None
            self.right_child = None
            return

        # Makes a decision on the splitting based on category
        if self.points.types[best_feature_id] == FeaturesTypes.REAL:
            left_mask = self.points.features[:, best_feature_id] <= value
            right_mask = self.points.features[:, best_feature_id] > value
        else:
            left_mask = self.points.features[:, best_feature_id] == value
            right_mask = self.points.features[:, best_feature_id] != value

        # Split the features based on the mask already created
        left_features = self.points.features[left_mask]
        right_features = self.points.features[right_mask]

        # Takes the labels corresponding to the features
        left_labels = self.points.labels[left_mask]
        right_labels = self.points.labels[right_mask]

        if len(left_features) < self.min_split_points or len(right_features) < self.min_split_points:
            self.left_child = None
            self.right_child = None
            return

        self.left_child = Tree(left_features, left_labels,
                               self.points.types, h - 1, self.min_split_points)
        self.right_child = Tree(
            right_features, right_labels, self.points.types, h - 1, self.min_split_points)

    def decide(self, features: List[float]) -> bool:
        """Give the guessed label of the tree to an unlabeled point

        Parameters
        ----------
            features : List[float]
                The features of the unlabeled point.

        Returns
        -------
            bool
                The label of the unlabeled point,
                guessed by the Tree
        """
        # If it's a leaf node, it returns the most common label among the labels in that node.
        if self.left_child is None and self.right_child is None:

            return np.argmax(np.bincount(self.points.labels)) > 0

        # If we are still in the tree find the best feature and category to split on
        best_feature_id = self.points.feature_id
        value = self.points.get_best_threshold()
        if value is None:
            value = 0

        type_real = self.points.types[best_feature_id] == FeaturesTypes.REAL
        left_branch_condition = (features[best_feature_id] <= value) if type_real else (
            features[best_feature_id] == value)

        if left_branch_condition:
            return self.left_child.decide(features)
        else:
            return self.right_child.decide(features)
        return False

    def print_tree(self, depth=0, is_left=None):
        indent = "    " * depth  # Indentation for each level
        branch = "──| " if is_left else "|── " if is_left is not None else ""

        if self.left_child is None and self.right_child is None:
            label = np.argmax(np.bincount(self.points.labels.astype(int)))
            print(f"{indent}{branch}Leaf: Predicted Label = {label}")
            # print(f"{indent}{branch}Labels = {self.points.labels}")

            return

        best_feature_id = self.points.feature_id
        feature_type = self.points.types[best_feature_id]

        category = self.points.get_best_threshold()
        if category is None:
            category = 0

        node_description = f"Feature {best_feature_id} <= {category}" if feature_type == FeaturesTypes.REAL else f"Feature {best_feature_id} == {category}"
        print(f"{indent}{branch}Node: {node_description}")

        if self.left_child is not None:
            self.left_child.print_tree(depth + 1, is_left=True)
        if self.right_child is not None:
            self.right_child.print_tree(depth + 1, is_left=False)
