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

        # if the integer part is large -> convert to the digits form
        # if the integer part is small -> pass as is
        # add "and" between the integer part and the numerator (this gramp to use with large numerators)
        optional_integer_part_default = pynini.closure(
            small_cardinal.optional_minus_graph
            + pynutil.insert("integer_part: \"")
            + (
                pynutil.add_weight(pynini.compose(small_cardinal.filter, single_digits_graph), -0.1)
                | at_least_one_digit
            )
            + pynutil.insert(" and\"")
            + pynini.accep(" "),
            0,
            1,
        ).optimize()

        optional_integer_part_for_small_num = pynini.closure(
            small_cardinal.optional_minus_graph
            + pynutil.insert("integer_part: \"")
            + (
                pynutil.add_weight(
                    pynini.compose(small_cardinal.filter, single_digits_graph) + pynutil.insert(" and\""), -0.1
                )
                | at_least_one_digit + pynutil.insert("\"")
            )
            + pynini.accep(" "),
            0,
            1,
        ).optimize()

        # # pass the integer part without changes, do NOT add "and" between the integer part and the numerator
        # optional_integer_part_small = pynini.closure(
        #     small_cardinal.optional_minus_graph
        #     + pynutil.insert("integer_part: \"")
        #     + at_least_one_digit
        #     + pynini.accep(" "),
        # )

        slash = pynini.cross("/", " slash\" ") | pynini.cross(" / ", " slash\" ")
        large_numerator = (
            pynutil.insert("numerator: \"") + pynini.compose(small_cardinal.filter, single_digits_graph) + slash
        )
        large_denominator = (
            pynutil.insert("denominator: \"")
            + pynutil.add_weight(pynini.compose(small_cardinal.filter, single_digits_graph), -0.1)
            + pynutil.insert("\"")
        )
        small_numerator = pynutil.insert("numerator: \"") + at_least_one_digit
        small_denominator = pynutil.insert("denominator: \"") + at_least_one_digit + pynutil.insert("\"")

        graph = optional_integer_part_default + large_numerator + large_denominator
        graph |= optional_integer_part_default + large_numerator + small_denominator
        graph |= optional_integer_part_default + small_numerator + slash + large_denominator

        # accept without transformations
        graph |= optional_integer_part_for_small_num + pynutil.add_weight(
            pynutil.insert("numerator: \"")
            + at_least_one_digit
            + (pynini.accep("/") | pynini.accep(" / "))
            + at_least_one_digit
            + pynutil.insert("\"")
            + pynutil.insert(" denominator: \"NONE\""),
            0.1,
        )

        graph = self.add_tokens(graph)
        self.fst = graph.optimize()

        # from pynini.lib.rewrite import top_rewrite
        # import pdb
        #
        # pdb.set_trace()
        # print(top_rewrite("1 12/124", graph))
        # print()
