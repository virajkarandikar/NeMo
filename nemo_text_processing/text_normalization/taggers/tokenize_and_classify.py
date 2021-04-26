# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.graph_utils import GraphFst, delete_extra_space, delete_space
from nemo_text_processing.text_normalization.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.taggers.date import DateFst
from nemo_text_processing.text_normalization.taggers.decimal import DecimalFst
from nemo_text_processing.text_normalization.taggers.measure import MeasureFst
from nemo_text_processing.text_normalization.taggers.money import MoneyFst
from nemo_text_processing.text_normalization.taggers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.taggers.punctuation import PunctuationFst
from nemo_text_processing.text_normalization.taggers.telephone import TelephoneFst
from nemo_text_processing.text_normalization.taggers.time import TimeFst
from nemo_text_processing.text_normalization.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.taggers.word import WordFst

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence including punctuation.
    For deployment, this grammar will be compiled and exported to OpenFst Finate State Archiv (FAR) File. 
    More details to deployment at NeMo/tools/text_processing_deployment.
    
    Args:
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, input_case: str):
        super().__init__(name="tokenize_and_classify", kind="classify")

        cardinal = CardinalFst()
        cardinal_graph = cardinal.fst

        ordinal = OrdinalFst(cardinal)
        ordinal_graph = ordinal.fst

        decimal = DecimalFst(cardinal)
        decimal_graph = decimal.fst

        measure_graph = MeasureFst(cardinal=cardinal, decimal=decimal).fst
        date_graph = DateFst(cardinal).fst
        word_graph = WordFst().fst
        time_graph = TimeFst().fst
        telephone_graph = TelephoneFst().fst
        money_graph = MoneyFst(cardinal=cardinal, decimal=decimal).fst
        whitelist_graph = WhiteListFst(input_case=input_case).fst
        punct_graph = PunctuationFst().fst

        classify = (
            pynutil.add_weight(whitelist_graph, 1.01)
            | pynutil.add_weight(time_graph, 1.1)
            | pynutil.add_weight(date_graph, 1.09)
            | pynutil.add_weight(decimal_graph, 1.1)
            | pynutil.add_weight(measure_graph, 1.1)
            | pynutil.add_weight(cardinal_graph, 1.1)
            | pynutil.add_weight(ordinal_graph, 1.1)
            | pynutil.add_weight(money_graph, 1.1)
            | pynutil.add_weight(telephone_graph, 1.1)
            | pynutil.add_weight(word_graph, 100)
        ).optimize()

        punct = pynutil.insert("tokens { ") + pynutil.add_weight(punct_graph, weight=1.1) + pynutil.insert(" }")
        token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
        token_plus_punct = (
            pynini.closure(punct + pynutil.insert(" ")) + token + pynini.closure(pynutil.insert(" ") + punct)
        )

        graph = token_plus_punct + pynini.closure(delete_extra_space + token_plus_punct)
        graph = delete_space + graph + delete_space

        self.fst = graph.optimize()
