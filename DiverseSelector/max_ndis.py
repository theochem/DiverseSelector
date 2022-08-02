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

from math import log, ceil
from pathlib import PurePath
from typing import Union

from DiverseSelector.base import SelectionBase
from DiverseSelector.utils import PandasDataFrame
import numpy as np
import random as rd


class ExtendedSelection(SelectionBase):
    """Dissimilarity based diversity subset selection."""

    def __init__(self,
                 features: Union[np.ndarray, PandasDataFrame, str, PurePath] = None,
                 arr_dist: np.ndarray = None,
                 normalize_features: bool = False,
                 sep: str = ",",
                 engine: str = "python",
                 initialization="medoid",
                 random_seed=42,
                 num_selected: int = None,
                 n_ary="JT",
                 weight="nw",
                 w_factor="fraction",
                 c_threshold=None,
                 algorithm='ECS_MeDiv'
                 ** kwargs,
                 ):
        """Initialization brute_strength_type for DissimilaritySelection class.
        Parameters
        ----------
        initialization
        metric
        random_seed
        feature_type
        mol_file
        feature_file
        num_selected
        arr_dist
        brute_strength_type
        r
        k
        cells
        max_dim
        grid_method
        kwargs
        """

        super().__init__(features,
                         arr_dist,
                         num_selected,
                         normalize_features,
                         sep,
                         engine,
                         random_seed,
                         **kwargs,
                         )
        self.initialization = initialization
        self.n_ary = n_ary
        self.weight = weight
        self.w_factor = w_factor
        self.c_threshold = c_threshold
        self.algorithm = algorithm
        self.n_total = len(self.features)
        # super(DissimilaritySelection, self).__init__(**kwargs)
        # self.__dict__.update(kwargs)

        # the initial compound index
        self.starting_idx = self.pick_initial_compounds()

    def pick_initial_compounds(self):
        """Pick the initial compounds."""

        # use the molecule with maximum distance to initial medoid as  the starting molecule
        if self.initialization == "medoid" or self.initialization == "outlier":
            starting_idx = calculate_special_points(total_data=self.features,
                                                    points=[self.initialization],
                                                    n_ary=self.n_ary,
                                                    c_threshold=self.c_threshold,
                                                    w_factor=self.w_factor,
                                                    weight=self.weight)

        elif self.initialization.lower() == "random":
            rng = np.random.default_rng(seed=self.random_seed)
            starting_idx = rng.choice(np.arange(self.features.shape[0]), 1)
        else:
            raise ValueError(f"Initialization method {self.initialization} is not supported.")

        return starting_idx

    def select(self):
        def _get_single_index(total_data, indices, selected_n, c_threshold=None, n_ary='RR',
                              weight='nw'):
            """Binary tie-breaker selection criterion"""

            index = len(total_data[0]) + 1
            min_value = 3.08
            for i in indices:
                v = 0
                for j in selected_n:
                    c_total = total_data[j] + total_data[i]
                    data_sets = [np.append(c_total, 2)]
                    Indices = gen_sim_dict(data_sets, c_threshold=c_threshold)
                    sim_index = Indices[weight][n_ary]
                    v += sim_index
                av_v = v / (len(selected_n) + 1)
                if av_v < min_value:
                    index = i
                    min_value = av_v
            return index

        def _get_new_index_n(total_data, selected_condensed, n, select_from_n, selected_n,
                             c_threshold=None,
                             n_ary='RR', weight='nw'):
            """Select a diverse object using the ECS_MeDiv algorithm"""
            n_total = n + 1
            # min value that is guaranteed to be higher than all the comparisons
            min_value = 3.08

            # placeholder index
            indices = [len(total_data[0]) + 1]

            # for all indices that have not been selected
            for i in select_from_n:
                # column sum
                c_total = selected_condensed + total_data[i]
                # calculating similarity
                data_sets = [np.append(c_total, n_total)]
                Indices = gen_sim_dict(data_sets, c_threshold=c_threshold)
                sim_index = Indices[weight][n_ary]
                # if the sim of the set is less than the similarity of the previous diverse set, update min_value and index
                if sim_index < min_value:
                    indices = [i]
                    min_value = sim_index
                elif sim_index == min_value:
                    indices.append(i)
            if len(indices) == 1:
                index = indices[0]
            else:
                if self.algorithm == 'ECS_MeDiv':
                    # Use average of binary similarities as tie-breaker
                    index = _get_single_index(total_data, indices, selected_n, c_threshold=None,
                                              n_ary=n_ary, weight='nw')
                elif self.algorithm == 'Max_nDis':
                    index = rd.choice(indices)
            return index

        total_indices = np.array(range(self.n_total))
        selected = [self.starting_idx]
        # vector with the column sums of all the selected fingerprints
        selected_condensed = self.features[seed]

        # number of fingerprints selected
        n = 1
        while len(selected) < self.num_selected:
            # indices from which to select the new fingerprints
            select_from_n = np.delete(total_indices, selected)

            # new index selected
            new_index_n = _get_new_index_n(self.features, selected_condensed, n, select_from_n,
                                           selected,
                                           c_threshold=self.c_threshold, n_ary=self.n_ary)

            # updating column sum vector
            selected_condensed += self.features[new_index_n]

            # updating selected indices
            selected.append(new_index_n)

            # updating n
            n = len(selected)

        return selected


def calculate_counters(data_sets, c_threshold=None, w_factor="fraction"):
    """Calculate 1-similarity, 0-similarity, and dissimilarity counters

    Arguments
    ---------
    data_sets : np.ndarray
        If it is a 1D array: Contains m + 1 elements,
        with m being the length of the fingerprints. The first
        m elements are the column sums of the matrix of fingerprints/features.
        The last element is the number of fingerprints.
        If it is a 2D array: Matrix of fingerprints/features.

    c_threshold : {None, 'dissimilar', int, float}
        Coincidence threshold.
        None : Default, c_threshold = n_objects % 2
        'dissimilar' : c_threshold = ceil(n_objects / 2)
        int : Integer number < n_objects
        float: Real number in the (0, 1) interval, c_threshold *= n_objects

    w_factor : {"fraction", "power_n"}
        Type of weight function that will be used.
        'fraction' : similarity = d[k]/n
                     dissimilarity = 1 - (d[k] - n_objects % 2)/n_objects
        'power_n' : similarity = n**-(n_objects - d[k])
                    dissimilarity = n**-(d[k] - n_objects % 2)
        other values : similarity = dissimilarity = 1

    Returns
    -------
    counters : dict
        Dictionary with the weighted and non-weighted counters.
    """
    # Setting matches
    if len(data_sets.shape) == 1:
        n_objects = int(data_sets[-1])
        c_total = data_sets[:-1]
    elif len(data_sets.shape) == 2:
        n_objects = len(data_sets)
        c_total = np.sum(data_sets, axis=0)
    else:
        raise TypeError("data_sets can only be a 1D or 2D Numpy array.")

    # Assign c_threshold
    if not c_threshold or c_threshold == 'min':
        c_threshold = n_objects % 2
    if isinstance(c_threshold, str):
        if c_threshold != 'dissimilar':
            raise TypeError(
                "c_threshold must be None, 'dissimilar', an integer, or a number in the (0, 1) interval.")
        else:
            c_threshold = ceil(n_objects / 2)
    if isinstance(c_threshold, int):
        if c_threshold >= n_objects:
            raise ValueError("c_threshold cannot be equal or greater than n_objects.")
        c_threshold = c_threshold
    if 0 < c_threshold < 1:
        c_threshold *= n_objects

    # Set w_factor
    if w_factor:
        if "power" in w_factor:
            power = int(w_factor.split("_")[-1])

            def f_s(d):
                return power ** -float(n_objects - d)

            def f_d(d):
                return power ** -float(d - n_objects % 2)
        elif w_factor == "fraction":
            def f_s(d):
                return d / n_objects

            def f_d(d):
                return 1 - (d - n_objects % 2) / n_objects
        else:
            def f_s(d):
                return 1

            def f_d(d):
                return 1
    else:
        def f_s(d):
            return 1

        def f_d(d):
            return 1

    # Calculate a, d, b + c
    a = 0
    w_a = 0
    d = 0
    w_d = 0
    total_dis = 0
    total_w_dis = 0
    for s in c_total:
        if 2 * s - n_objects > c_threshold:
            a += 1
            w_a += f_s(2 * s - n_objects)
        elif n_objects - 2 * s > c_threshold:
            d += 1
            w_d += f_s(abs(2 * s - n_objects))
        else:
            total_dis += 1
            total_w_dis += f_d(abs(2 * s - n_objects))
    total_sim = a + d
    total_w_sim = w_a + w_d
    p = total_sim + total_dis
    w_p = total_w_sim + total_w_dis

    counters = {"a": a, "w_a": w_a, "d": d, "w_d": w_d,
                "total_sim": total_sim, "total_w_sim": total_w_sim,
                "total_dis": total_dis, "total_w_dis": total_w_dis,
                "p": p, "w_p": w_p}

    return counters


def gen_sim_dict(data_sets, c_threshold=None, w_factor="fraction"):
    counters = calculate_counters(data_sets, c_threshold=c_threshold, w_factor="fraction")
    # Indices
    # AC: Austin-Colwell, BUB: Baroni-Urbani-Buser, CTn: Consoni-Todschini n
    # Fai: Faith, Gle: Gleason, Ja: Jaccard, Ja0: Jaccard 0-variant
    # JT: Jaccard-Tanimoto, RT: Rogers-Tanimoto, RR: Russel-Rao
    # SM: Sokal-Michener, SSn: Sokal-Sneath n

    # Weighted Indices
    ac_w = (2 / np.pi) * np.arcsin(np.sqrt(counters['total_w_sim'] /
                                           counters['w_p']))
    bub_w = ((counters['w_a'] * counters['w_d']) ** 0.5 + counters['w_a']) / \
            ((counters['w_a'] * counters['w_d']) ** 0.5 + counters['w_a'] + counters['total_w_dis'])
    ct1_w = (log(1 + counters['w_a'] + counters['w_d'])) / \
            (log(1 + counters['w_p']))
    ct2_w = (log(1 + counters['w_p']) - log(1 + counters['total_w_dis'])) / \
            (log(1 + counters['w_p']))
    ct3_w = (log(1 + counters['w_a'])) / \
            (log(1 + counters['w_p']))
    ct4_w = (log(1 + counters['w_a'])) / \
            (log(1 + counters['w_a'] + counters['total_w_dis']))
    fai_w = (counters['w_a'] + 0.5 * counters['w_d']) / \
            (counters['w_p'])
    gle_w = (2 * counters['w_a']) / \
            (2 * counters['w_a'] + counters['total_w_dis'])
    ja_w = (3 * counters['w_a']) / \
           (3 * counters['w_a'] + counters['total_w_dis'])
    ja0_w = (3 * counters['total_w_sim']) / \
            (3 * counters['total_w_sim'] + counters['total_w_dis'])
    jt_w = (counters['w_a']) / \
           (counters['w_a'] + counters['total_w_dis'])
    rt_w = (counters['total_w_sim']) / \
           (counters['w_p'] + counters['total_w_dis'])
    rr_w = (counters['w_a']) / \
           (counters['w_p'])
    sm_w = (counters['total_w_sim']) / \
           (counters['w_p'])
    ss1_w = (counters['w_a']) / \
            (counters['w_a'] + 2 * counters['total_w_dis'])
    ss2_w = (2 * counters['total_w_sim']) / \
            (counters['w_p'] + counters['total_w_sim'])

    ## Non-Weighted Indices
    ac_nw = (2 / np.pi) * np.arcsin(np.sqrt(counters['total_w_sim'] /
                                            counters['p']))
    bub_nw = ((counters['w_a'] * counters['w_d']) ** 0.5 + counters['w_a']) / \
             ((counters['a'] * counters['d']) ** 0.5 + counters['a'] + counters['total_dis'])
    ct1_nw = (log(1 + counters['w_a'] + counters['w_d'])) / \
             (log(1 + counters['p']))
    ct2_nw = (log(1 + counters['w_p']) - log(1 + counters['total_w_dis'])) / \
             (log(1 + counters['p']))
    ct3_nw = (log(1 + counters['w_a'])) / \
             (log(1 + counters['p']))
    ct4_nw = (log(1 + counters['w_a'])) / \
             (log(1 + counters['a'] + counters['total_dis']))
    fai_nw = (counters['w_a'] + 0.5 * counters['w_d']) / \
             (counters['p'])
    gle_nw = (2 * counters['w_a']) / \
             (2 * counters['a'] + counters['total_dis'])
    ja_nw = (3 * counters['w_a']) / \
            (3 * counters['a'] + counters['total_dis'])
    ja0_nw = (3 * counters['total_w_sim']) / \
             (3 * counters['total_sim'] + counters['total_dis'])
    jt_nw = (counters['w_a']) / \
            (counters['a'] + counters['total_dis'])
    rt_nw = (counters['total_w_sim']) / \
            (counters['p'] + counters['total_dis'])
    rr_nw = (counters['w_a']) / \
            (counters['p'])
    sm_nw = (counters['total_w_sim']) / \
            (counters['p'])
    ss1_nw = (counters['w_a']) / \
             (counters['a'] + 2 * counters['total_dis'])
    ss2_nw = (2 * counters['total_w_sim']) / \
             (counters['p'] + counters['total_sim'])

    # Dictionary with all the results
    Indices = {'nw': {'AC': ac_nw,
                      'BUB': bub_nw,
                      'CT1': ct1_nw,
                      'CT2': ct2_nw,
                      'CT3': ct3_nw,
                      'CT4': ct4_nw,
                      'Fai': fai_nw,
                      'Gle': gle_nw,
                      'Ja0': ja0_nw,
                      'Ja': ja_nw,
                      'JT': jt_nw,
                      'RT': rt_nw,
                      'RR': rr_nw,
                      'SM': sm_nw,
                      'SS1': ss1_nw,
                      'SS2': ss2_nw},
               'w': {'AC': ac_w,
                     'BUB': bub_w,
                     'CT1': ct1_w,
                     'CT2': ct2_w,
                     'CT3': ct3_w,
                     'CT4': ct4_w,
                     'Fai': fai_w,
                     'Gle': gle_w,
                     'Ja0': ja0_w,
                     'Ja': ja_w,
                     'JT': jt_w,
                     'RT': rt_w,
                     'RR': rr_w,
                     'SM': sm_w,
                     'SS1': ss1_w,
                     'SS2': ss2_w}}
    return Indices


def calculate_special_points(total_data, points=['medoid'],
                             complete_order=False, n_ary='RR',
                             c_threshold=None, w_factor='fraction', weight='nw'):
    """Calculate medoid of a set"""
    total_sum = np.sum(total_data, axis=0)
    n_objects = len(total_data)
    complementary_sims = np.array([])
    for i in range(n_objects):
        i_sum = total_sum - total_data[i]
        data_sets = np.append(i_sum, n_objects - 1)
        Indices = gen_sim_dict(data_sets, c_threshold=c_threshold, w_factor=w_factor)
        sim_index = Indices[weight][n_ary]
        complementary_sims = np.append(complementary_sims, sim_index)
    special_points = {}
    if 'medoid' in points:
        special_points['medoid'] = np.argmin(complementary_sims)
    if 'outlier' in points:
        special_points['outlier'] = np.argmax(complementary_sims)
    if 'complete_order' in points:
        special_points['complete_order'] = np.argsort(complementary_sims)
    return special_points
