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

"""Selectors classes for different choices of subset selection."""
import collections
import math
from typing import Union

import bitarray
from DiverseSelector.base import KDTreeBase, SelectionBase
from DiverseSelector.diversity import compute_diversity
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class MaxMin(SelectionBase):
    """Selecting compounds using MinMax algorithm.

    Initial point is chosen as medoid center. The second point is
    the furthest point away. All the following points are selected
    using the rule:
    1. Find minimum distance from every point to the selected ones.
    2. Select a point the has the maximum distance among calculated
       on the previous step.
    """

    def __init__(self, func_distance=None):
        """
        Initializing class.

        Parameters
        ----------
        func_distance: callable
            Function for calculating the pairwise distance between instances of the array.
        """
        self.func_distance = func_distance

    def select_from_cluster(self, arr, num_selected, cluster_ids=None):
        """
        Algorithm MinMax for selecting points from cluster.

        Parameters
        ----------
        arr: np.ndarray
            Distance matrix for points that needs to be selected if func_distance is None.
            Otherwise, treated as coordinates array.
        num_selected: int
            Number of molecules that need to be selected
        cluster_ids: np.ndarray
            Indices of molecules that form a cluster

        Returns
        -------
        selected: list
            List of ids of selected molecules
        """
        if self.func_distance is not None:
            arr_dist = self.func_distance(arr)
        else:
            arr_dist = arr

        if cluster_ids is not None:
            arr_dist = arr_dist[cluster_ids][:, cluster_ids]

        # choosing initial point as the medoid
        selected = [np.argmin(np.sum(arr_dist, axis=0))]
        while len(selected) < num_selected:
            min_distances = np.min(arr_dist[selected], axis=0)
            new_id = np.argmax(min_distances)
            selected.append(new_id)
        return selected


class MaxSum(SelectionBase):
    """Selecting compounds using MaxSum algorithm.

    Initial point is chosen as medoid center. The second point is
    the furthest point away. All the following points are selected
    using the rule:
    1. Find minimum distance from every point to the selected ones.
    2. Select a point the has the maximum sum of distance among calculated
       on the previous step.
    """

    def __init__(self, func_distance=None):
        """
        Initializing class.

        Parameters
        ----------
        func_distance: callable
            Function for calculating the pairwise distance between instances of the array.
        """
        self.func_distance = func_distance

    def select_from_cluster(self, arr, num_selected, cluster_ids=None):
        """
        Algorithm MinMax for selecting points from cluster.

        Parameters
        ----------
        arr: np.ndarray
            Distance matrix for points that needs to be selected if func_distance is None.
            Otherwise, treated as coordinates array.
        num_selected: int
            Number of molecules that need to be selected
        cluster_ids: np.ndarray
            Indices of molecules that form a cluster

        Returns
        -------
        selected: list
            List of ids of selected molecules
        """
        if num_selected > len(arr):
            raise ValueError(f"Requested {num_selected} points which is greater than {len(arr)} "
                             f"points provided in array")

        if self.func_distance is not None:
            arr_dist = self.func_distance(arr)
        else:
            arr_dist = arr

        if cluster_ids is not None:
            arr_dist = arr_dist[cluster_ids][:, cluster_ids]

        # choosing initial point as the medoid
        selected = [np.argmin(np.sum(arr_dist, axis=0))]
        while len(selected) < num_selected:
            sum_distances = np.sum(arr_dist[selected], axis=0)
            while True:
                new_id = np.argmax(sum_distances)
                if new_id in selected:
                    sum_distances[new_id] = 0
                else:
                    break
            selected.append(new_id)
        return selected


class OptiSim(SelectionBase):
    """Selecting compounds using OptiSim algorithm.

    Initial point is chosen as medoid center. Points are randomly chosen and added to a subsample
    if outside of radius r from all previously selected points, and discarded otherwise. Once k
    number of points are added to the subsample, the point with the greatest minimum distance to
    previously selected points is selected and the subsample is cleared and the process repeats.

    Addapted from  https://doi.org/10.1021/ci970282v
    """

    def __init__(
            self,
            r=None,
            k=10,
            tolerance=5.0,
            func_distance=lambda x, y: np.linalg.norm(x - y),
            start_id=0,
            random_seed=42,
    ):
        """
        Initializing class.

        Parameters
        ----------
        tolerance: float
            Percentage error of number of molecules actually selected from number of molecules
            requested.
        r: float
            Initial guess of radius for optisim algorithm, no points within r distance to an already
            selected point can be selected.
        k: int
            Amount of points to add to subsample before selecting one of the points with the
            greatest minimum distance
            to the previously selected points.
        func_distance: callable
            Function for calculating the pairwise distance between instances of the array.
        start_id: int
            Index for the first point to be selected.
        random_seed: int
            Seed for random selection of points be evaluated.
        """
        self.r = r
        self.k = k
        self.tolerance = tolerance
        self.func_distance = func_distance
        self.start_id = start_id
        self.random_seed = random_seed

    def algorithm(self, arr) -> list:
        """
        Optisim algorithm logic.

        Parameters
        ----------
        arr: np.ndarray
            Coordinate array of points.

        Returns
        -------
        selected: list
            List of ids of selected molecules
        """
        selected = [self.start_id]
        recycling = []

        candidates = np.delete(np.arange(0, len(arr)), selected + recycling)
        subsample = {}
        while len(candidates) > 0:
            while len(subsample) < self.k:
                if len(candidates) == 0:
                    if len(subsample) > 0:
                        break
                    return selected
                rng = np.random.default_rng(seed=self.random_seed)
                random_int = rng.integers(low=0, high=len(candidates), size=1)[0]
                index_new = candidates[random_int]
                distances = []
                for selected_idx in selected:
                    data_point = arr[index_new]
                    selected_point = arr[selected_idx]
                    distance = self.func_distance(selected_point, data_point)
                    distances.append(distance)
                min_dist = min(distances)
                if min_dist > self.r:
                    subsample[index_new] = min_dist
                else:
                    recycling.append(index_new)
                candidates = np.delete(
                    np.arange(0, len(arr)),
                    selected + recycling + list(subsample.keys()),
                )
            selected.append(max(zip(subsample.values(), subsample.keys()))[1])
            candidates = np.delete(np.arange(0, len(arr)), selected + recycling)
            subsample = {}

        return selected

    def select_from_cluster(self, arr, num_selected, cluster_ids=None):
        """
        Algorithm that uses optisim for selecting points from cluster.

        Parameters
        ----------
        arr: np.ndarray
            Coordinate array of points
        num_selected: int
            Number of molecules that need to be selected.
        cluster_ids: np.ndarray
            Indices of molecules that form a cluster

        Returns
        -------
        selected: list
            List of ids of selected molecules
        """
        return predict_radius(self, arr, num_selected, cluster_ids)


class DirectedSphereExclusion(KDTreeBase):
    """Selecting points using Directed Sphere Exclusion algorithm.

    Starting point is chosen as the reference point and not included in the selected molecules. The
    distance of each point is calculated to the reference point and the points are then sorted based
    on the ascending order of distances. The points are then evaluated in their sorted order, and
    are selected if their distance to all the other selected points is at least r away. Euclidian
    distance is used by default and the r value is automatically generated if not passed to satisfy
    the number of molecules requested.

    Adapted from https://doi.org/10.1021/ci025554v
    """

    def __init__(self, r=None, tolerance=5.0, func_distance=lambda x, y: np.linalg.norm(x - y),
                 start_id=0, random_seed=42):
        """
        Initializing class.

        Parameters
        ----------
        r: float
            Initial guess of radius for directed sphere exclusion algorithm, no points within r
            distance to an already selected point can be selected.
        tolerance: float
            Percentage error of number of molecules actually selected from number of molecules
            requested.
        func_distance: callable
            Function for calculating the pairwise distance between instances of the array.
        start_id: int
            Index for the first point to be selected.
        random_seed: int
            Seed for random selection of points be evaluated.
        """
        self.r = r
        self.tolerance = tolerance
        self.func_distance = func_distance
        self.starting_idx = start_id
        self.random_seed = random_seed
        self.BT = collections.namedtuple("BT", ["value", "index", "left", "right"])

    def algorithm(self, arr):
        """
        Directed sphere exclusion algorithm logic.

        Parameters
        ----------
        arr: np.ndarray
            Coordinate array of points.

        Returns
        -------
        selected: list
            List of ids of selected molecules
        """
        selected = []
        candidates = np.delete(np.arange(0, len(arr)), self.starting_idx)
        distances = []
        for idx in candidates:
            ref_point = arr[self.starting_idx]
            data_point = arr[idx]
            distance = self.func_distance(ref_point, data_point)
            distances.append((distance, idx))
        distances.sort()
        order = [idx for dist, idx in distances]

        kdtree = self._kdtree(arr)
        bv = bitarray.bitarray(len(arr))
        bv[:] = 0
        bv[self.starting_idx] = 1

        for idx in order:
            if not bv[idx]:
                selected.append(idx)
                elim = self._find_nearest_neighbor(kdtree=kdtree, point=arr[idx], threshold=self.r)
                for index in elim:
                    bv[index] = 1

        return selected

    def select_from_cluster(self, arr, num_selected, cluster_ids=None):
        """
        Algorithm that uses sphere_exclusion for selecting points from cluster.

        Parameters
        ----------
        arr: np.ndarray
            Coordinate array of points
        num_selected: int
            Number of molecules that need to be selected.
        cluster_ids: np.ndarray
            Indices of molecules that form a cluster

        Returns
        -------
        selected: list
            List of ids of selected molecules
        """
        return predict_radius(self, arr, num_selected, cluster_ids)


class GridPartitioning(SelectionBase):
    """Selecting points using the Grid Partitioning algorithm.

    Points are partitioned into grids using an algorithm (equisized independent or equisized
    dependent). A point is selected from each of the grids while the number of selected points is
    less than the number requested and while the grid has available points remaining, looping until
    the number of requested points is satisfied. If at the end, the number of points needed is less
    than the number of grids available to select from, the points are chosen from the grids with the
    greatest diversity.

    Adapted from https://doi.org/10.1016/S1093-3263(99)00016-9.
    """

    def __init__(self, cells, grid_method="equisized_independent", max_dim=None, random_seed=42):
        """
        Initializing class.

        Parameters
        ----------
        cells: int
            Number of cells to partition each axis into, the number of resulting grids is cells to
            the power of the dimensionality of the coordinate array.
        grid_method: str
            Grid method used to partition the points into grids. "equisized_independent" and
            "equisized_dependent" are supported options.
        max_dim: int
            Maximum dimensionality of coordinate array, if the dimensionality is greater than the
            max_dim provided then dimensionality reduction is done using PCA.
        random_seed: int
            Seed for random selection of points to be selected from each grid.
        """
        self.random_seed = random_seed
        self.cells = cells
        self.max_dim = max_dim
        self.grid_method = grid_method

    def select_from_cluster(self, arr, num_selected, cluster_ids=None):
        """
        Grid partitioning algorithm for selecting points from cluster.

        Parameters
        ----------
        arr: np.ndarray
            Coordinate array of points
        num_selected: int
            Number of molecules that need to be selected.
        cluster_ids: np.ndarray
            Indices of molecules that form a cluster

        Returns
        -------
        selected: list
            List of ids of selected molecules
        """
        if cluster_ids is not None:
            arr = arr[cluster_ids]

        selected = []
        data_dim = len(arr[0])
        if self.max_dim is not None and data_dim > self.max_dim:
            norm_data = StandardScaler().fit_transform(arr)
            pca = PCA(n_components=self.max_dim)
            arr = pca.fit_transform(norm_data)
            data_dim = self.max_dim

        if self.grid_method == "equisized_independent":
            axis_info = []
            for i in range(data_dim):
                axis_min, axis_max = min(arr[:, i]), max(arr[:, i])
                cell_length = (axis_max - axis_min) / self.cells
                axis_info.append([axis_min, axis_max, cell_length])
            bins = {}
            for index, point in enumerate(arr):
                point_bin = []
                for dim, value in enumerate(point):
                    if value == axis_info[dim][0]:
                        index_bin = 0
                    elif value == axis_info[dim][1]:
                        index_bin = self.cells - 1
                    else:
                        index_bin = int((value - axis_info[dim][0]) // axis_info[dim][2])
                    point_bin.append(index_bin)
                bins.setdefault(tuple(point_bin), [])
                bins[tuple(point_bin)].append(index)

        elif self.grid_method == "equisized_dependent":
            bins = {}
            for i in range(data_dim):
                if len(bins) == 0:
                    axis_min, axis_max = min(arr[:, i]), max(arr[:, i])
                    cell_length = (axis_max - axis_min) / self.cells
                    axis_info = [axis_min, axis_max, cell_length]

                    for index, point in enumerate(arr):
                        point_bin = []
                        if point[i] == axis_info[0]:
                            index_bin = 0
                        elif point[i] == axis_info[1]:
                            index_bin = self.cells - 1
                        else:
                            index_bin = int((point[i] - axis_info[0]) // axis_info[2])
                        point_bin.append(index_bin)
                        bins.setdefault(tuple(point_bin), [])
                        bins[tuple(point_bin)].append(index)
                else:
                    new_bins = {}
                    for bin_idx, bin_list in bins.items():
                        axis_min = min(arr[bin_list, i])
                        axis_max = max(arr[bin_list, i])
                        cell_length = (axis_max - axis_min) / self.cells
                        axis_info = [axis_min, axis_max, cell_length]

                        for point_idx in bin_list:
                            point_bin = [num for num in bin_idx]
                            if arr[point_idx][i] == axis_info[0]:
                                index_bin = 0
                            elif arr[point_idx][i] == axis_info[1]:
                                index_bin = self.cells - 1
                            else:
                                index_bin = int((arr[point_idx][i] - axis_info[0]) // axis_info[2])
                            point_bin.append(index_bin)
                            new_bins.setdefault(tuple(point_bin), [])
                            new_bins[tuple(point_bin)].append(point_idx)
                    bins = new_bins

        elif self.grid_method == "equifrequent_independent":
            raise NotImplementedError(f"{self.grid_method} not implemented.")
        elif self.grid_method == "equifrequent_dependent":
            raise NotImplementedError(f"{self.grid_method} not implemented.")
        else:
            raise ValueError(f"{self.grid_method} not a valid grid_method")

        old_len = 0
        to_delete = []
        rng = np.random.default_rng(seed=self.random_seed)
        while len(selected) < num_selected:
            num_needed = num_selected - len(selected)
            bin_count = len(bins)
            if bin_count <= num_needed:
                for bin_idx, bin_list in bins.items():
                    random_int = rng.integers(low=0, high=len(bin_list), size=1)[0]
                    mol_id = bin_list.pop(random_int)
                    selected.append(mol_id)
                    if len(bin_list) == 0:
                        to_delete.append(bin_idx)
                for idx in to_delete:
                    del bins[idx]
                to_delete = []

            else:
                diversity = []
                for bin_idx, bin_list in bins.items():
                    diversity.append((compute_diversity(arr[bin_list]), bin_idx))
                diversity.sort(reverse=True)
                for _, bin_idx in diversity[:num_needed]:
                    random_int = rng.integers(low=0, high=len(bins[bin_idx]), size=1)[0]
                    mol_id = bins[bin_idx].pop(random_int)
                    selected.append(mol_id)
            if len(selected) == old_len:
                break
            old_len = len(selected)
        return selected


class KDTree(KDTreeBase):
    """Selecting points using an algorithm adapted from KDTree.

    Points are initially used to construct a KDTree. Eucleidean distances are used for this
    algorithm. The first point selected is based on the starting_idx provided and becomes the first
    query point. An approximation of the furthest point to the query point is found using
    find_furthest_neighbor and is selected. find_nearest_neighbor is then done to eliminate close
    neighbors to the new selected point. Medoid is then calculated from previously selected points
    and is used as the new query point for find_furthest_neighbor, repeating the process. Terminates
    upon selecting requested number of points or if all available points exhausted.

    Adapted from: https://en.wikipedia.org/wiki/K-d_tree#Construction
    """

    def __init__(self,
                 start_id=0,
                 func_distance=lambda x, y: sum((i - j) ** 2 for i, j in zip(x, y)),
                 scaling=10,
                 ):
        """
        Initializing class.

        Parameters
        ----------
        start_id: int
            Index for the first point to be selected.
        func_distance: callable
            Function for calculating the pairwise distance between instances of the array.
        scaling: float
            Percent of average maximum distance to use when eliminating the closest points.
        """

        self.starting_idx = start_id
        self.func_distance = func_distance
        self.BT = collections.namedtuple("BT", ["value", "index", "left", "right"])
        self.FNRecord = collections.namedtuple("FNRecord", ["point", "index", "distance"])
        self.scaling = scaling / 100
        self.ratio = None

    def _find_furthest_neighbor(self, kdtree, point, selected_bitvector):
        """Find approximately the furthest neighbor in a k-d tree for a given point.

        Parameters
        ----------
        kdtree: collections.namedtuple
            KDTree organizing coordinates.
        point: list
            Query point for search.
        selected_bitvector: bitarray
            Bitvector to keep track of previously selected points from array.

        Returns
        -------
        best: collections.namedtuple
            The furthest point found in search.
        """

        k = len(point)
        best = None

        def search(tree, depth):
            # Recursively search through the k-d tree to find the
            # furthest neighbor.

            nonlocal selected_bitvector
            nonlocal best

            if tree is None:
                return

            if not selected_bitvector[tree.index]:
                distance = self.func_distance(tree.value, point)
                if best is None or distance > best.distance:
                    best = self.FNRecord(point=tree.value, index=tree.index, distance=distance)

            axis = depth % k
            diff = point[axis] - tree.value[axis]
            if diff <= 0:
                close, away = tree.left, tree.right
            else:
                close, away = tree.right, tree.left

            search(tree=away, depth=depth + 1)
            if best is None or (close is not None and diff ** 2 <= 1.1 * (
                    (point[axis] - close.value[axis]) ** 2)):
                search(tree=close, depth=depth + 1)

        search(tree=kdtree, depth=0)
        return best

    def select_from_cluster(self, arr, num_selected, cluster_ids=None):
        """Main function for selecting points using the KDTree algorithm.

        Parameters
        ----------
        arr: np.ndarray
            Coordinate array of points
        num_selected: int
            Number of molecules that need to be selected.
        cluster_ids: np.ndarray
            Indices of molecules that form a cluster

        Returns
        -------
        selected: list
            List of ids of selected molecules
        """
        if cluster_ids is not None:
            arr = arr[cluster_ids]

        if isinstance(arr, np.ndarray):
            arr = arr.tolist()
        arr_len = len(arr)
        tree = self._kdtree(arr)

        bv = bitarray.bitarray(arr_len)
        bv[:] = 0
        selected = [self.starting_idx]
        query_point = arr[self.starting_idx]
        bv[self.starting_idx] = 1
        count = 1
        num_eliminate = arr_len - num_selected
        self.ratio = math.ceil(num_eliminate / num_selected)
        best_distance_av = 0
        while len(selected) < num_selected:
            new_point = self._find_furthest_neighbor(tree, query_point, bv)
            if new_point is None:
                return selected
            selected.append(new_point.index)
            bv[new_point.index] = 1
            query_point = (count * np.array(query_point) + np.array(new_point.point)) / (count + 1)
            query_point = query_point.tolist()
            if count == 1:
                best_distance_av = new_point.distance
            else:
                best_distance_av = (count * best_distance_av + new_point.distance) / (count + 1)
            if count == 1:
                if num_eliminate > 0 and self.scaling != 0:
                    num_eliminate = self._eliminate(tree, arr[self.starting_idx],
                                                    best_distance_av * self.scaling,
                                                    num_eliminate, bv)
            if num_eliminate > 0 and self.scaling != 0:
                num_eliminate = self._eliminate(tree, new_point.point,
                                                best_distance_av * self.scaling,
                                                num_eliminate, bv)
            count += 1
        return selected


def predict_radius(obj: Union[DirectedSphereExclusion, OptiSim], arr, num_selected,
                   cluster_ids=None):
    """
    Algorithm that uses sphere_exclusion for selecting points from cluster.

    Parameters
    ----------
    obj: object
        Instance of dissimilarity selection class
    arr: np.ndarray
        Coordinate array of points
    num_selected: int
        Number of molecules that need to be selected.
    cluster_ids: np.ndarray
        Indices of molecules that form a cluster

    Returns
    -------
    selected: list
        List of ids of selected molecules
    """
    if not isinstance(obj, (DirectedSphereExclusion, OptiSim)):
        raise ValueError("Not valid class for function.")

    if cluster_ids is not None:
        arr = arr[cluster_ids]

    original_r = None
    if obj.r is not None:
        original_r = obj.r
        result = obj.algorithm(arr)
    # Use numpy.optimize.bisect instead
    else:
        rg = max(np.ptp(arr, axis=0)) / num_selected * 3
        obj.r = rg
        result = obj.algorithm(arr)
    if len(result) == num_selected:
        return result

    low = obj.r if len(result) > num_selected else 0
    high = obj.r if low == 0 else None
    bounds = [low, high]
    count = 0
    error = num_selected * obj.tolerance / 100
    while (len(result) < num_selected - error or len(result) > num_selected + error) and count < 20:
        if bounds[1] is None:
            rg = bounds[0] * 2
        else:
            rg = (bounds[0] + bounds[1]) / 2
        obj.r = rg
        result = obj.algorithm(arr)
        if len(result) > num_selected:
            bounds[0] = rg
        else:
            bounds[1] = rg
        count += 1
    if count == 10:
        print(f"Optimal radius finder failed to converge, selected {len(result)} molecules instead "
              f"of requested {num_selected}.")
    obj.r = original_r
    return result
