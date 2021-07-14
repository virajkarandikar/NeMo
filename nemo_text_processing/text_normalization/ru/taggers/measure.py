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
    NEMO_NON_BREAKING_SPACE,
    NEMO_SIGMA,
    GraphFst,
    delete_space,
    get_abs_path,
)
from nemo_text_processing.text_normalization.ru.alphabet import RU_ALPHA

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure, suppletive aware, e.g. 
        -12kg -> measure { negative: "true" cardinal { integer: "twelve" } units: "kilograms" }
        1kg -> measure { cardinal { integer: "one" } units: "kilogram" }
        .5kg -> measure { decimal { fractional_part: "five" } units: "kilograms" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        fraction: FractionFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.cardinal_numbers

        # if not deterministic:
        #     cardinal_graph |= cardinal.range_graph

        graph_unit = pynini.string_file(get_abs_path("ru/data/measurements.tsv"))

        # TODO add vocab for singular/plural
        graph_unit_plural = graph_unit
        optional_graph_negative = cardinal.optional_graph_negative

        graph_unit2 = pynini.cross("/", "в") + delete_space + pynutil.insert(NEMO_NON_BREAKING_SPACE) + graph_unit

        optional_graph_unit2 = pynini.closure(
            delete_space + pynutil.insert(NEMO_NON_BREAKING_SPACE) + graph_unit2, 0, 1,
        )

        unit_plural = (
            pynutil.insert("units: \"")
            + (graph_unit_plural + optional_graph_unit2 | graph_unit2)
            + pynutil.insert("\"")
        )

        unit_singular = (
            pynutil.insert("units: \"") + (graph_unit + optional_graph_unit2 | graph_unit2) + pynutil.insert("\"")
        )

        subgraph_decimal = (
            # pynutil.insert("decimal { ")
            decimal.final_graph
            + delete_space
            # + pynutil.insert(" } ")
            + unit_plural
        )

        subgraph_cardinal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + ((NEMO_SIGMA - "1") @ cardinal_graph)
            + delete_space
            + pynutil.insert("\"")
            + pynutil.insert(" } ")
            + unit_plural
        )

        # TODO one -> fix for Ru
        subgraph_cardinal |= (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + pynini.cross("1", "one")
            + delete_space
            + pynutil.insert("\"")
            + pynutil.insert(" } ")
            + unit_singular
        )

        cardinal_dash_alpha = (
            pynutil.insert("cardinal { integer: \"")
            + cardinal_graph
            + pynini.cross('-', '')
            + pynutil.insert("\" } units: \"")
            + pynini.closure(RU_ALPHA, 1)
            + pynutil.insert("\"")
        )

        alpha_dash_cardinal = (
            pynutil.insert("units: \"")
            + pynini.closure(RU_ALPHA, 1)
            + pynini.cross('-', '')
            + pynutil.insert("\"")
            + pynutil.insert(" cardinal { integer: \"")
            + cardinal_graph
            + pynutil.insert("\" } preserve_order: true")
        )

        decimal_dash_alpha = (
            # pynutil.insert("decimal { ")
            decimal.final_graph
            + pynini.cross('-', '')
            + pynutil.insert(" units: \"")
            + pynini.closure(RU_ALPHA, 1)
            + pynutil.insert("\"")
        )

        alpha_dash_decimal = (
            pynutil.insert("units: \"")
            + pynini.closure(RU_ALPHA, 1)
            + pynini.cross('-', '')
            + pynutil.insert("\"")
            + pynutil.insert(" decimal { ")
            + decimal.final_graph
            + pynutil.insert(" } preserve_order: true")
        )

        # subgraph_fraction = (
        #     pynutil.insert("fraction { ") + fraction.graph + delete_space + pynutil.insert(" } ") + unit_plural
        # )

        final_graph = (
            subgraph_decimal
            | subgraph_cardinal
            | cardinal_dash_alpha
            | alpha_dash_cardinal
            | decimal_dash_alpha
            | alpha_dash_decimal
            # | subgraph_fraction
        )
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
