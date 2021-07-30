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
    NEMO_SIGMA,
    PLURAL_TO_SINGULAR,
    SINGULAR_TO_PLURAL,
    GraphFst,
    convert_space,
    insert_space,
)
from nemo_text_processing.text_normalization.en.taggers.date import get_hundreds_graph
from nemo_text_processing.text_normalization.en.utils import get_abs_path

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


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

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph
        graph_decimal_final = decimal.final_graph_wo_negative

        unit_singular = pynini.string_file(get_abs_path("data/currency/currency.tsv"))
        unit_plural = convert_space(unit_singular @ SINGULAR_TO_PLURAL)
        unit_singular = convert_space(unit_singular)

        graph_unit_singular = pynutil.insert("currency: \"") + unit_singular + pynutil.insert("\"")
        graph_unit_plural = pynutil.insert("currency: \"") + unit_plural + pynutil.insert("\"")

        singular_graph = (
            graph_unit_singular + pynutil.insert(" integer_part: \"") + pynini.cross("1", "one") + pynutil.insert("\"")
        )

        graph_decimal = graph_unit_plural + insert_space + graph_decimal_final

        if deterministic:
            graph_integer = (
                graph_unit_plural
                + pynutil.insert(" integer_part: \"")
                + ((NEMO_SIGMA - "1") @ cardinal_graph)
                + pynutil.insert("\"")
            )
        else:
            graph_integer = (
                graph_unit_plural
                + pynutil.insert(" integer_part: \"")
                + ((NEMO_SIGMA - "1") @ (get_hundreds_graph(deterministic) | cardinal_graph))
                + pynutil.insert("\"")
            )
            graph_decimal |= singular_graph + insert_space + graph_decimal_final

        graph_integer |= singular_graph

        final_graph = graph_integer | graph_decimal

        if not deterministic:
            # currency = pynini.project(graph_unit_plural, "input")
            minor_singular = pynini.string_file(get_abs_path("data/currency/currency_minor_singular.tsv"))
            minor_plural = pynini.string_file(get_abs_path("data/currency/currency_minor_plural.tsv"))

            currency = pynutil.delete("currency: \"") + NEMO_SIGMA + pynutil.delete("\"") + pynutil.delete(NEMO_SIGMA)
            currency = pynini.compose(graph_decimal, currency)
            currency_maj = pynini.compose(currency, PLURAL_TO_SINGULAR)
            currency_min_sing = pynini.compose(currency_maj, minor_singular)
            currency_min_plural = pynini.compose(currency_maj, minor_plural)

            frac_one = NEMO_SIGMA + pynini.accep("fractional_part: \"one\"") + NEMO_SIGMA
            frac_non_one = pynini.difference(NEMO_SIGMA, frac_one)
            graph_decimal_with_minor_currency_plural = pynini.compose(graph_decimal, frac_non_one) + pynutil.insert(
                " currency_minor: \"" + currency_min_plural + pynutil.insert("\"")
            )
            graph_decimal_with_minor_currency_singular = pynini.compose(graph_decimal, frac_one) + pynutil.insert(
                " currency_minor: \"" + currency_min_sing + pynutil.insert("\"")
            )
            final_graph |= graph_decimal_with_minor_currency_singular
            final_graph |= graph_decimal_with_minor_currency_plural

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

        from pynini.lib.rewrite import top_rewrites
        import pdb

        pdb.set_trace()
        print()
