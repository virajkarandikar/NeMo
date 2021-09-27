# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, GraphFst

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
    "23 4/5" ->
    tokens { fraction { integer: "twenty three" numerator: "four" denominator: "five" } }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, small_cardinal, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        single_digits_graph = small_cardinal.single_digits_graph
        at_least_one_digit = pynini.closure(NEMO_DIGIT, 1)
        # large integer part
        self.filter = (
            small_cardinal.optional_minus
            + small_cardinal.filter
            + pynini.accep(" ")
            + at_least_one_digit
            + (pynini.accep("/") | pynini.accep(" / "))
            + at_least_one_digit
        )
        # large numerator
        self.filter |= (
            pynini.closure(at_least_one_digit + pynini.accep(" "), 0, 1)
            + small_cardinal.filter
            + (pynini.accep("/") | pynini.accep(" / "))
            + at_least_one_digit
        )
        # large denominator
        self.filter |= (
            pynini.closure(at_least_one_digit + pynini.accep(" "), 0, 1)
            + at_least_one_digit
            + (pynini.accep("/") | pynini.accep(" / "))
            + small_cardinal.filter
        )
        integer = pynutil.insert("integer_part: \"") + single_digits_graph + pynutil.insert("\"") + pynini.accep(" ")
        numerator = (
            pynutil.insert("numerator: \"")
            + single_digits_graph
            + (pynini.cross("/", " by\" ") | pynini.cross(" / ", "by\" "))
        )
        denominator = pynutil.insert("denominator: \"") + small_cardinal.single_digits_graph + pynutil.insert("\"")

        graph = pynini.closure(integer, 0, 1) + numerator + denominator
        graph = pynini.compose(self.filter, graph.optimize()).optimize()
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
