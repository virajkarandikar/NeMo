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
from pynini.lib.rewrite import top_rewrite

try:
    import pynini
    from pynini.lib import pynutil

    delete_space = pynutil.delete(" ")

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g. 
        -12.5006 billion -> decimal { negative: "true" integer_part: "12"  fractional_part: "five o o six" quantity: "billion" }
        1 billion -> decimal { integer_part: "one" quantity: "billion" }

    cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        large_integer_part = (
            cardinal.optional_minus_graph + pynutil.insert("integer_part: \"") + cardinal.graph + pynutil.insert("\"")
        )
        small_integer_part = (
            pynutil.insert("integer_part: \"")
            + pynini.closure(pynini.accep("-"), 0, 1)
            + NEMO_DIGIT ** (0, 4)
            + pynutil.insert("\"")
        )
        integer_part = pynutil.add_weight(large_integer_part, -100) | small_integer_part

        # large_fractional_part = pynini.cross(".", "") + pynutil.insert("fractional_part: \"") + pynini.compose(NEMO_DIGIT ** (4, ...), cardinal.single_digits_graph) + pynutil.insert("\"")
        # small_fractional_part = pynini.cross(".", "") + pynutil.insert("fractional_part: \"") + pynini.closure(NEMO_DIGIT ** (0, 3)) + pynutil.insert("\"")
        # fractional_part = pynutil.add_weight(large_fractional_part, -100) | small_fractional_part

        fractional_part = (
            pynini.cross(".", "")
            + pynutil.insert("fractional_part: \"")
            + cardinal.single_digits_graph
            + pynutil.insert("\"")
        )

        self.final_graph = pynini.closure(integer_part + pynutil.insert(" "), 0, 1) + fractional_part
        self.fst = self.add_tokens(self.final_graph.optimize())
