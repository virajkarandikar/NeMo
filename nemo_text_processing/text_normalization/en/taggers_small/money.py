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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    SINGULAR_TO_PLURAL,
    GraphFst,
    convert_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.en.taggers.date import get_hundreds_graph
from nemo_text_processing.text_normalization.en.utils import get_abs_path, load_labels

try:
    import pynini
    from pynini.lib import pynutil, rewrite

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


# class MoneyFst(GraphFst):
#     """
#     Finite state transducer for classifying money, suppletive aware, e.g.
#         $12.05 -> money { currency: "dollars" integer_part: "twelve" fractional_part: "o five" }
#         $1 -> money { currency: "dollar" integer_part: "one" }
#
#     Args:
#         cardinal: CardinalFst
#         decimal: DecimalFst
#         deterministic: if True will provide a single transduction option,
#             for False multiple transduction are generated (used for audio-based normalization)
#     """
#
#     def __init__(
#         self,
#         default_cardinal: GraphFst,
#         default_decimal: GraphFst,
#         small_cardinal: GraphFst,
#         small_decimal: GraphFst,
#         deterministic: bool = True,
#     ):
#         super().__init__(name="money", kind="classify", deterministic=deterministic)
#
#         default_money = defaultMoneyFst(cardinal=default_cardinal, decimal=default_decimal)
#         filter = (
#             pynini.project(default_money.currency_unit, "input")
#             + pynini.closure(pynini.accep(" "), 0, 1)
#             + (small_decimal.filter | small_cardinal.filter)
#         )
#         self.fst = pynini.compose(filter, default_money.fst)
#
#         assert top_rewrite("123.01891", graph) == 'integer_part: "one two three" fractional_part: "zero one eight nine one"'


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, suppletive aware, e.g.
        $12.05 -> money { currency: "dollars" integer_part: "twelve" fractional_part: "o five" }
        $1 -> money { currency: "dollar" integer_part: "one" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(
        self, small_cardinal: GraphFst, small_decimal: GraphFst, deterministic: bool = True,
    ):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        single_digits_graph = small_cardinal.single_digits_graph

        unit_singular_raw = pynini.string_file(get_abs_path("data/currency/currency.tsv"))
        unit_plural = convert_space(unit_singular_raw @ SINGULAR_TO_PLURAL)
        unit_singular = convert_space(unit_singular_raw)
        self.currency_unit = (unit_singular | unit_singular_raw).optimize()

        graph_unit_singular = pynutil.insert("currency: \"") + unit_singular + pynutil.insert("\"")
        graph_unit_plural = pynutil.insert("currency: \"") + unit_plural + pynutil.insert("\"")

        space = insert_space | pynini.accep(" ")

        graph_decimal = graph_unit_plural + space + small_decimal.final_graph
        graph_integer_large = (
            graph_unit_plural
            + space
            + pynutil.insert("integer_part: \"")
            + ((NEMO_SIGMA - "1") @ single_digits_graph)
            + pynutil.insert("\"")
        )

        currency = pynini.project(unit_singular_raw, "input") + pynini.closure(pynini.accep(" "), 0, 1)
        graph_integer_large = pynini.compose(currency + small_cardinal.filter, graph_integer_large)
        #
        # self.filter = (
        #     pynini.project(unit_singular_raw, "input")
        #     + pynini.closure(pynini.accep(" "), 0, 1)
        #     + (small_cardinal.filter + pynini.closure(pynini.accep(".") + NEMO_DIGIT**(4,...), 0, 1))
        # )

        final_graph = graph_integer_large | graph_decimal
        # final_graph = pynini.compose(self.filter, final_graph.optimize()).optimize()
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

        # from pynini.lib.rewrite import top_rewrite
        # import pdb;
        # pdb.set_trace()
        # print()
