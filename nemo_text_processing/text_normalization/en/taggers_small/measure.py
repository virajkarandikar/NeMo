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
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_NON_BREAKING_SPACE,
    NEMO_SIGMA,
    NEMO_SPACE,
    SINGULAR_TO_PLURAL,
    GraphFst,
    convert_space,
    delete_space,
)
from nemo_text_processing.text_normalization.en.taggers.measure import MeasureFst as defaultMeasureFst
from nemo_text_processing.text_normalization.en.taggers.ordinal import OrdinalFst as OrdinalTagger
from nemo_text_processing.text_normalization.en.utils import get_abs_path
from nemo_text_processing.text_normalization.en.verbalizers.ordinal import OrdinalFst as OrdinalVerbalizer

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


# class MeasureFst(GraphFst):
#     """
#     Finite state transducer for classifying measure, suppletive aware, e.g.
#         -12kg -> measure { negative: "true" cardinal { integer: "twelve" } units: "kilograms" }
#         1kg -> measure { cardinal { integer: "one" } units: "kilogram" }
#         .5kg -> measure { decimal { fractional_part: "five" } units: "kilograms" }
#
#     Args:
#         cardinal: CardinalFst
#         decimal: DecimalFst
#         fraction: FractionFst
#         deterministic: if True will provide a single transduction option,
#             for False multiple transduction are generated (used for audio-based normalization)
#     """
#
#     def __init__(self, cardinal: GraphFst, decimal: GraphFst, fraction: GraphFst, deterministic: bool = True):
#         super().__init__(name="measure", kind="classify", deterministic=deterministic)
#
#         default_measure = defaultMeasureFst(cardinal=cardinal, decimal=decimal, fraction=fraction)
#
#         # add constraint to large fractions?
#         filter = pynini.union(
#             NEMO_DIGIT ** (5, ...) + NEMO_SIGMA,
#             pynini.closure(NEMO_DIGIT) + pynini.accep(".") + NEMO_DIGIT ** (4, ...) + NEMO_SIGMA,
#             NEMO_DIGIT ** (5, ...) + pynini.accep(".") + pynini.closure(NEMO_DIGIT, 1) + NEMO_SIGMA,
#         )
#
#         self.fst = pynini.compose(filter, default_measure.fst.optimize()).optimize()


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

    def __init__(
        self, small_cardinal: GraphFst, small_decimal: GraphFst, fraction: GraphFst, deterministic: bool = True
    ):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)
        single_digits_graph = small_cardinal.single_digits_graph

        graph_unit_raw = pynini.string_file(get_abs_path("data/measurements.tsv"))
        graph_unit_plural = convert_space(graph_unit_raw @ SINGULAR_TO_PLURAL)
        graph_unit = convert_space(graph_unit_raw)
        optional_graph_negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)
        graph_unit2 = pynini.cross("/", "per") + delete_space + pynutil.insert(NEMO_NON_BREAKING_SPACE) + graph_unit
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
            pynutil.insert("decimal { ")
            + small_decimal.final_graph
            + delete_space
            + pynutil.insert(" } ")
            + unit_plural
        )

        subgraph_cardinal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + ((NEMO_SIGMA - "1") @ single_digits_graph)
            + delete_space
            + pynutil.insert("\"")
            + pynutil.insert(" } ")
            + unit_plural
        )

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

        # TODO add fraction
        # subgraph_fraction = (
        #     pynutil.insert("fraction { ") + fraction.graph + delete_space + pynutil.insert(" } ") + unit_plural
        # )

        filter_cardinal = (
            small_cardinal.filter + pynini.closure(pynini.accep(" "), 0, 1) + pynini.project(graph_unit_raw, "input")
        )

        final_graph = (
            subgraph_decimal
            | pynini.compose(filter_cardinal, subgraph_cardinal)
            # | subgraph_fraction
        ).optimize()

        # final_graph = pynini.compose(self.filter, final_graph)
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
