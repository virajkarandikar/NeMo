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

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst
from nemo_text_processing.text_normalization.en.taggers.money import MoneyFst as defaultMoneyFst

try:
    import pynini
    from pynini.lib import pynutil, rewrite

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

    def __init__(
        self,
        default_cardinal: GraphFst,
        default_decimal: GraphFst,
        small_cardinal: GraphFst,
        small_decimal: GraphFst,
        deterministic: bool = True,
    ):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        default_money = defaultMoneyFst(cardinal=default_cardinal, decimal=default_decimal)
        filter = (
            pynini.project(default_money.currency_unit, "input")
            + pynini.closure(pynini.accep(" "), 0, 1)
            + (small_decimal.filter | small_cardinal.filter)
        )
        self.fst = pynini.compose(filter, default_money.fst)
