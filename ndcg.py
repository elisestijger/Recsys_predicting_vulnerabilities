# Copyright 2016 Krysta M Bouzek
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np

"""
Implementation of normalized discounted cumulative gain.

Handy for testing ranking algorithms.

https://en.wikipedia.org/wiki/Discounted_cumulative_gain
"""

def cum_gain(relevance):
    """
    Calculate cumulative gain.
    This ignores the position of a result, but may still be generally useful.

    @param relevance: Graded relevances of the results.
    @type relevance: C{seq} or C{numpy.array}
    """

    if relevance is None or len(relevance) < 1:
        return 0.0

    return np.asarray(relevance).sum()


def dcg(relevance, alternate=True):
    """
    Calculate discounted cumulative gain.

    @param relevance: Graded and ordered relevances of the results.
    @type relevance: C{seq} or C{numpy.array}
    @param alternate: True to use the alternate scoring (intended to
    place more emphasis on relevant results).
    @type alternate: C{bool}
    """

    if relevance is None or len(relevance) < 1:
        return 0.0

    rel = np.asarray(relevance)
    p = len(rel)
    # print("real", rel)
    if alternate:
        # from wikipedia: "An alternative formulation of
        # DCG[5] places stronger emphasis on retrieving relevant documents"

        log2i = np.log2(np.asarray(range(1, p + 1)) + 1)
        return ((np.power(2, rel) - 1) / log2i).sum()
    else:
        log2i = np.log2(range(2, p + 1))
        return rel[0] + (rel[1:] / log2i).sum()


def idcg(relevance, alternate=True):
    """
    Calculate ideal discounted cumulative gain (maximum possible DCG).

    @param relevance: Graded and ordered relevances of the results.
    @type relevance: C{seq} or C{numpy.array}
    @param alternate: True to use the alternate scoring (intended to
    place more emphasis on relevant results).
    @type alternate: C{bool}
    """

    if relevance is None or len(relevance) < 1:
        return 0.0

    # guard copy before sort
    rel = np.asarray(relevance).copy()
    rel = sorted(rel)
    # print("ideal", rel)
    return dcg(rel[::-1], alternate)


def ndcg(relevance, nranks, alternate=False):

    if relevance is None or len(relevance) < 1:
        return 0.0

    if (nranks < 1):
        raise Exception('nranks < 1')

    rel = np.asarray(relevance)
    pad = max(0, nranks - len(rel))

    # pad could be zero in which case this will no-op
    rel = np.pad(rel, (0, pad), 'constant')

    # now slice downto nranks
    rel = rel[0:min(nranks, len(rel))]

    ideal_dcg = idcg(rel, alternate)
    
    # print("ideal", ideal_dcg)
    if ideal_dcg == 0:
        return 0.0
    actual = dcg(rel, alternate)

    return actual / ideal_dcg
