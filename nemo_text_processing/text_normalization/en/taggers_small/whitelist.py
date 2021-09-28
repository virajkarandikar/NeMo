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

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, convert_space
from nemo_text_processing.text_normalization.en.taggers.whitelist import WhiteListFst as defaultWhiteListFst
from nemo_text_processing.text_normalization.en.utils import get_abs_path, load_labels

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelist, e.g.
        misses -> tokens { name: "mrs" }
        for non-deterministic case: "Dr. Abc" ->
            tokens { name: "drive" } tokens { name: "Abc" }
            tokens { name: "doctor" } tokens { name: "Abc" }
            tokens { name: "Dr." } tokens { name: "Abc" }
    This class has highest priority among all classifier grammars. Whitelisted tokens are defined and loaded from "data/whitelist.tsv".

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, input_case: str, deterministic: bool = True):
        super().__init__(name="whitelist", kind="classify", deterministic=deterministic)

        default_whitelist = defaultWhiteListFst(input_case=input_case)
        whitelist_accepted = default_whitelist.whitelist_graph | default_whitelist.graph

        measure_units = pynini.string_file(get_abs_path("data/measurements.tsv"))
        currency = pynini.string_file(get_abs_path("data/currency/currency.tsv"))

        # exclude words that are measure units or currency symbols
        to_exclude = pynini.union(pynini.project(measure_units, "input"), pynini.project(currency, "input"))
        filter = pynini.difference(pynini.project(whitelist_accepted, "input"), to_exclude).optimize()

        self.fst = pynini.compose(filter, default_whitelist.fst).optimize()
