# -*- coding: utf-8 -*-
# The DiverseSelector library provides a set of tools to select molecule
# subset with maximum molecular diversity.
#
# Copyright (C) 2022 The QC-Devs Community
#
# This file is part of DiverseSelector.
#
# DiverseSelector is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# DiverseSelector is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --

"""Base class for diversity based subset selection."""
from abc import ABC, abstractmethod
import collections
import warnings

import numpy as np


class SelectionBase(ABC):
    """Base class for subset selection."""

    def select(self, arr, num_selected, labels=None):
        """
         Algorithm for selecting points.

        Parameters
        ----------
        arr: np.ndarray
            Array of features if fun_distance is provided.
            Otherwise, treated as distance matrix.
        num_selected: int
            Number of points that need to be selected
        labels: np.ndarray
            Labels for performing algorithm withing clusters.

        Returns
        -------
        selected: list
            List of ids of selected molecules
        """
        if labels is None:
            return self.select_from_cluster(arr, num_selected)

        # compute the number of samples (i.e. population or pop) in each cluster
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels)
        pop_clusters = {unique_label: len(np.where(labels == unique_label)[0])
                        for unique_label in unique_labels}
        # compute number of samples to be selected from each cluster
        n = num_selected // num_clusters

        # update number of samples to select from each cluster based on the cluster population.
        # this is needed when some clusters do not have enough samples in them (pop < n) and
        # needs to be done iteratively until all remaining clusters have at least n samples
        selected_ids = []
        while np.any([value <= n for value in pop_clusters.values() if value != 0]):
            for unique_label in unique_labels:
                if pop_clusters[unique_label] != 0:
                    # get index of sample labelled with unique_label
                    cluster_ids = np.where(labels == unique_label)[0]
                    if len(cluster_ids) <= n:
                        # all samples in the cluster are selected & population becomes zero
                        selected_ids.append(cluster_ids)
                        pop_clusters[unique_label] = 0
            # update number of samples to be selected from each cluster
            totally_used_clusters = list(pop_clusters.values()).count(0)
            n = (num_selected - len(np.hstack(selected_ids))) // \
                (num_clusters - totally_used_clusters)

            warnings.warn(
                f"Number of molecules in one cluster is less than"
                f" {num_selected}/{num_clusters}.\nNumber of selected "
                f"molecules might be less than desired.\nIn order to avoid this "
                f"problem. Try to use less number of clusters"
            )

        for unique_label in unique_labels:
            if pop_clusters[unique_label] != 0:
                # sample n ids from cluster labeled unique_label
                cluster_ids = np.where(labels == unique_label)[0]
                selected = self.select_from_cluster(arr, n, cluster_ids)
                selected_ids.append(cluster_ids[selected])

        return np.hstack(selected_ids).flatten().tolist()

    @abstractmethod
    def select_from_cluster(self, arr, num_selected, cluster_ids=None):
        """
        Algorithm for selecting points from cluster.

        Parameters
        ----------
        arr: np.ndarray
            Distance matrix for points that needs to be selected
        num_selected: int
            Number of molecules that need to be selected
        cluster_ids: np.array


        Returns
        -------
        selected: list
            List of ids of molecules that are belonged to the one cluster

        """
        pass


class KDTreeBase(SelectionBase, ABC):
    """Base class for KDTree based subset selection."""

    def __int__(self):
        """Initializing class."""
        self.func_distance = lambda x, y: sum((i - j) ** 2 for i, j in zip(x, y))
        self.BT = collections.namedtuple("BT", ["value", "index", "left", "right"])

    def _kdtree(self, arr):
        """Construct a k-d tree from an iterable of points.

        Parameters
        ----------
        arr: list or np.ndarray
            Coordinate array of points.

        Returns
        -------
        kdtree: collections.namedtuple
            KDTree organizing coordinates.
        """

        k = len(arr[0])

        def build(points, depth, old_indices=None):
            """Build a k-d tree from a set of points at a given depth."""
            if len(points) == 0:
                return None
            middle = len(points) // 2
            indices, points = zip(*sorted(enumerate(points), key=lambda x: x[1][depth % k]))
            if old_indices is not None:
                indices = [old_indices[i] for i in indices]
            return self.BT(
                value=points[middle],
                index=indices[middle],
                left=build(
                    points=points[:middle],
                    depth=depth + 1,
                    old_indices=indices[:middle],
                ),
                right=build(
                    points=points[middle + 1:],
                    depth=depth + 1,
                    old_indices=indices[middle + 1:],
                ),
            )

        kdtree = build(points=arr, depth=0)
        return kdtree

    def _find_nearest_neighbor(self, kdtree, point, threshold):
        """
        Find the nearest neighbors in a k-d tree for a point.

        Parameters
        ----------
        kdtree: collections.namedtuple
            KDTree organizing coordinates.
        point: list
            Query point for search.
        threshold: float
            The boundary used to mark all the points whose distance is within the threshold.

        Returns
        -------
        to_eliminate: list
            A list containing all the indices of points too close to the newly selected point.
        """
        k = len(point)
        to_eliminate = []

        def search(tree, depth):
            # Recursively search through the k-d tree to find the
            # nearest neighbor.

            if tree is None:
                return

            distance = self.func_distance(tree.value, point)
            if distance < threshold:
                to_eliminate.append((distance, tree.index))

            axis = depth % k
            diff = point[axis] - tree.value[axis]
            if diff <= 0:
                close, away = tree.left, tree.right
            else:
                close, away = tree.right, tree.left

            search(tree=close, depth=depth + 1)
            if diff ** 2 < threshold:
                search(tree=away, depth=depth + 1)

        search(tree=kdtree, depth=0)
        to_eliminate.sort()
        to_eliminate.pop(0)
        to_eliminate = [index for dist, index in to_eliminate]
        return to_eliminate

    def _eliminate(self, tree, point, threshold, num_eliminate, bv):
        """Eliminates points from being selected in future rounds.

        Parameters
        ----------
        tree: collections.namedtuple
            KDTree organizing coordinates.
        point: list
            Point where close neighbors should be eliminated.
        threshold: float
            An average of all the furthest distances found using find_furthest_neighbor
        num_eliminate: int
            Maximum number of points permitted to be eliminated.
        bv: bitarray
            Bitvector marking picked/eliminated points.

        Returns
        -------
        num_eliminate: int
            Maximum number of points permitted to be eliminated.
        """
        elim_candidates = self._find_nearest_neighbor(tree, point, threshold)
        elim_candidates = elim_candidates[:self.ratio]
        num_eliminate -= len(elim_candidates)
        if num_eliminate < 0:
            elim_candidates = elim_candidates[:num_eliminate]
        for index in elim_candidates:
            bv[index] = 1
        return num_eliminate
