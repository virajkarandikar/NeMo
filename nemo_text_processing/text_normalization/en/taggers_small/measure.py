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

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, fraction: GraphFst, deterministic: bool = True):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)

        default_measure = defaultMeasureFst(cardinal=cardinal, decimal=decimal, fraction=fraction)

        # add constraint to large fractions?
        filter = pynini.union(
            NEMO_DIGIT ** (5, ...) + NEMO_SIGMA,
            pynini.closure(NEMO_DIGIT) + pynini.accep(".") + NEMO_DIGIT ** (4, ...) + NEMO_SIGMA,
            NEMO_DIGIT ** (5, ...) + pynini.accep(".") + pynini.closure(NEMO_DIGIT, 1) + NEMO_SIGMA,
        )

        self.fst = pynini.compose(filter, default_measure.fst.optimize()).optimize()
