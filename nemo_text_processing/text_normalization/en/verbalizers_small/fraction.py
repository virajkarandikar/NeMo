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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, insert_space
from nemo_text_processing.text_normalization.en.verbalizers.ordinal import OrdinalFst

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fraction
        e.g. tokens { fraction { integer: "twenty three" numerator: "four" denominator: "five" } } ->
        twenty three four fifth

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)

        integer = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\" ")
        numerator = pynutil.delete("numerator: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\" ")
        denominator = (
            insert_space + pynutil.delete("denominator: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        )
        denominator |= pynutil.add_weight(
            pynutil.delete("denominator: \"") + pynini.cross("NONE", "") + pynutil.delete("\""), -0.1
        )

        graph = pynini.closure(integer + insert_space, 0, 1) + numerator + denominator

        self.graph = graph
        delete_tokens = self.delete_tokens(self.graph)
        self.fst = delete_tokens.optimize()
