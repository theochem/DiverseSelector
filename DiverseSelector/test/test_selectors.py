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

"""Testing for the MaxMin selection algorithms."""

from DiverseSelector.selectors import DirectedSphereExclusion, GridPartitioning, KDTree,\
    MaxMin, MaxSum, OptiSim
from DiverseSelector.test.common import generate_synthetic_data
import numpy as np
from numpy.testing import assert_equal
from sklearn.metrics import pairwise_distances

coords, class_labels, arr_dist = generate_synthetic_data(n_samples=100,
                                                         n_features=2,
                                                         n_clusters=1,
                                                         pairwise_dist=True,
                                                         metric="euclidean",
                                                         random_state=42)

coords_cluster, class_labels_cluster, arr_dist_cluster = generate_synthetic_data(n_samples=100,
                                                                                 n_features=2,
                                                                                 n_clusters=3,
                                                                                 pairwise_dist=True,
                                                                                 metric="euclidean",
                                                                                 random_state=42)

simple_coords = np.array([[0, 0],
                          [2, 0],
                          [0, 2],
                          [2, 2],
                          [-10, -10]])


def test_maxmin():
    """Testing the MinMax class."""
    selector = MaxMin()
    selected_ids = selector.select(arr=arr_dist_cluster,
                                   num_selected=12,
                                   labels=class_labels_cluster)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [41, 34, 94, 85, 51, 50, 66, 78, 21, 64, 29, 83])

    selector = MaxMin()
    selected_ids = selector.select(arr=arr_dist, num_selected=12)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [85, 57, 41, 25, 9, 62, 29, 65, 81, 61, 60, 97])

    selector = MaxMin(lambda x: pairwise_distances(x, metric='euclidean'))
    selected_ids = selector.select(arr=simple_coords, num_selected=3)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [0, 4, 3])

    # generating mocked clusters
    np.random.seed(42)
    cluster_one = np.random.normal(0, 1, (3, 2))
    cluster_two = np.random.normal(10, 1, (6, 2))
    cluster_three = np.random.normal(20, 1, (10, 2))
    labels_mocked = np.hstack([[0 for i in range(3)],
                               [1 for i in range(6)],
                               [2 for i in range(10)]])

    mocked_cluster_coords = np.vstack([cluster_one, cluster_two, cluster_three])

    # selecting molecules
    selector = MaxMin(lambda x: pairwise_distances(x, metric='euclidean'))
    selected_mocked = selector.select(mocked_cluster_coords,
                                      num_selected=15,
                                      labels=labels_mocked)
    assert_equal(selected_mocked, [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 15, 10, 13, 9, 18])


def test_maxsum():
    """Testing OptiSim class."""
    selector = MaxSum(lambda x: pairwise_distances(x, metric='euclidean'))
    selected_ids = selector.select(arr=coords_cluster, num_selected=12, labels=class_labels_cluster)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [41, 34, 85, 94, 51, 50, 78, 66, 21, 64, 0, 83])

    selector = MaxSum(lambda x: pairwise_distances(x, metric='euclidean'))
    selected_ids = selector.select(arr=coords, num_selected=12)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [85, 57, 25, 41, 95, 9, 21, 8, 13, 68, 37, 54])


def test_optisim():
    """Testing OptiSim class."""
    selector = OptiSim()
    selected_ids = selector.select(arr=coords_cluster, num_selected=12, labels=class_labels_cluster)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [2, 85, 86, 59, 1, 50, 93, 68, 0, 11, 33, 46])

    selector = OptiSim()
    selected_ids = selector.select(arr=coords, num_selected=12)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [0, 13, 21, 9, 8, 18, 39, 57, 65, 25, 6, 45])


def test_directedsphereexclusion():
    """Testing DirectedSphereExclusion class."""
    selector = DirectedSphereExclusion()
    selected_ids = selector.select(arr=coords_cluster, num_selected=12, labels=class_labels_cluster)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [95, 14, 88, 84, 76, 49, 93, 50, 29, 27, 15, 54])

    selector = DirectedSphereExclusion()
    selected_ids = selector.select(arr=coords, num_selected=12)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [17, 92, 64, 6, 12, 76, 10, 87, 73, 66, 11, 57])


def test_gridpartitioning():
    """Testing DirectedSphereExclusion class."""
    selector = GridPartitioning(cells=3)
    selected_ids = selector.select(arr=coords_cluster, num_selected=12, labels=class_labels_cluster)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [2, 25, 84, 56, 8, 70, 58, 78, 4, 46, 65, 29])

    selector = GridPartitioning(cells=3)
    selected_ids = selector.select(arr=coords, num_selected=12)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [7, 55, 70, 57, 29, 91, 9, 65, 28, 11, 54, 88])


def test_kdtree():
    """Testing KDTree class."""
    selector = KDTree()
    selected_ids = selector.select(arr=coords_cluster, num_selected=12, labels=class_labels_cluster)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [2, 73, 94, 86, 1, 50, 93, 78, 0, 54, 33, 72])

    selector = KDTree()
    selected_ids = selector.select(arr=coords, num_selected=12)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [0, 95, 57, 41, 25, 9, 8, 6, 66, 1, 42, 82])
