from typing import List, Tuple

from enum import Enum
import numpy as np


class FeaturesTypes(Enum):
    """Enumerate possible features types"""
    BOOLEAN = 0
    CLASSES = 1
    REAL = 2


class PointSet:
    """A class representing set of training points.

    Attributes
    ----------
        types : List[FeaturesTypes]
            Each element of this list is the type of one of the
            features of each point
        features : np.array[float]
            2D array containing the features of the points. Each line
            corresponds to a point, each column to a feature.
        labels : np.array[bool]
            1D array containing the labels of the points.
    """

    def __init__(self, features: List[List[float]], labels: List[bool], types: List[FeaturesTypes]):
        """
        Parameters
        ----------
        features : List[List[float]]
            The features of the points. Each sublist contained in the
            list represents a point, each of its elements is a feature
            of the point. All the sublists should have the same size as
            the `types` parameter, and the list itself should have the
            same size as the `labels` parameter.
        labels : List[bool]
            The labels of the points.
        types : List[FeaturesTypes]
            The types of the features of the points.
        """
        self.types = types
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.feature_id = None
        self.best_value = None

    def get_gini(self) -> float:
        """Computes the Gini score of the set of points

        Returns
        -------
        float
            The Gini score of the set of points
        """

        return self.get_gini_subset(self.labels)

    def get_gini_subset(self, subset) -> float:
        """Computes the Gini score of a subset of points

        Returns
        -------
        float
            The Gini score of the set of points
        """
        if len(subset) == 0:
            return 0.0

        probabilities = (np.bincount(subset) / len(subset))
        gini = 1 - np.sum([p ** 2 for p in probabilities if p > 0])
        return gini

    def get_best_gain(self, min_split_points=1) -> Tuple[int, float]:
        """Compute the feature along which splitting provides the best gain

        Returns
        -------
        int
            The ID of the feature along which splitting the set provides the
            best Gini gain.
        float
            The best Gini gain achievable by splitting this set along one of
            its features.
        """
        best_feature_id = -1
        best_gini = 0.0
        best_value = None
        # Calcuate the best gini besed on each feature and each category
        for feature_id in range(self.features.shape[1]):
            if self.types[feature_id] == FeaturesTypes.REAL:
                # Finds the unique sorted potential thresholds
                thresholds = np.unique(self.features[:, feature_id])
                for thr in thresholds:
                    gini_gain = self.calculate_gini_gain(feature_id, thr)
                    if gini_gain > best_gini:
                        left_mask = self.features[:, feature_id] <= thr
                        right_mask = self.features[:, feature_id] > thr
                        if np.sum(left_mask) >= min_split_points and np.sum(right_mask) >= min_split_points:
                            best_gini = gini_gain
                            best_feature_id = feature_id
                            # Define L and R sets based on the current threshold
                            L = self.features[self.features[:,
                                                            feature_id] <= thr, feature_id]
                            R = self.features[self.features[:,
                                                            feature_id] > thr, feature_id]
                            best_value = (max(L) + min(R)) / 2
            else:
                if self.types[feature_id] == FeaturesTypes.BOOLEAN:
                    categories = [0]
                else:
                    categories = np.unique(self.features[:, feature_id])
                    if len(categories) == 2:
                        categories = [categories[0]]
                for category in categories:
                    gini_gain = self.calculate_gini_gain(feature_id, category)
                    if gini_gain > best_gini:
                        left_mask = self.features[:, feature_id] == category
                        right_mask = self.features[:, feature_id] != category
                        if np.sum(left_mask) >= min_split_points and np.sum(right_mask) >= min_split_points:
                            best_gini = gini_gain
                            best_value = category
                            best_feature_id = feature_id

        self.feature_id = best_feature_id
        self.best_value = best_value
        return best_feature_id, best_gini

    def calculate_gini_gain(self, feature_id: int, value: float) -> float:
        """Compute the gini gain provided the feature id and the value which can be either the category or the threshold
        depending if the feature type

        Returns
        -------
        float
            The  Gini gain achievable by splitting this set along the given feature and given categoty.
        """
        mask_condition = (self.features[:, feature_id] <= value) if self.types[feature_id] == FeaturesTypes.REAL else (
            self.features[:, feature_id] == value)
        left_mask = mask_condition
        right_mask = ~mask_condition

        left_subset = self.labels[left_mask]
        right_subset = self.labels[right_mask]

        left_gini = self.get_gini_subset(left_subset)
        right_gini = self.get_gini_subset(right_subset)

        total = len(self.labels)
        gini_gain = (len(left_subset) / total) * left_gini + \
            (len(right_subset) / total) * right_gini
        return self.get_gini() - gini_gain

    def get_best_threshold(self) -> float:
        if self.feature_id == None:
            raise Exception(
                "get_best_gain must be called before get_best_threshold")
        elif self.types[self.feature_id] == FeaturesTypes.BOOLEAN:
            return None
        else:
            return self.best_value
