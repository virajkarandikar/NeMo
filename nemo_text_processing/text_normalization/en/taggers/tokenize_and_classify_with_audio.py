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

import os

from nemo_text_processing.text_normalization.en.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
    get_abs_path,
)
from nemo_text_processing.text_normalization.en.taggers.abbreviation import AbbreviationFst
from nemo_text_processing.text_normalization.en.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.en.taggers.date import DateFst
from nemo_text_processing.text_normalization.en.taggers.decimal import DecimalFst
from nemo_text_processing.text_normalization.en.taggers.electronic import ElectronicFst
from nemo_text_processing.text_normalization.en.taggers.fraction import FractionFst
from nemo_text_processing.text_normalization.en.taggers.measure import MeasureFst
from nemo_text_processing.text_normalization.en.taggers.money import MoneyFst
from nemo_text_processing.text_normalization.en.taggers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.en.taggers.punctuation import PunctuationFst
from nemo_text_processing.text_normalization.en.taggers.roman import RomanFst
from nemo_text_processing.text_normalization.en.taggers.telephone import TelephoneFst
from nemo_text_processing.text_normalization.en.taggers.time import TimeFst
from nemo_text_processing.text_normalization.en.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.en.taggers.word import WordFst

from nemo.utils import logging

try:
    import pynini
    from pynini.lib import pynutil
    from tools.text_processing_deployment.pynini_export import _generator_main

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
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, input_case: str, deterministic: bool = True, use_cache: bool = False):
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)

        far_file = get_abs_path("_tokenize_and_classify_non_deteministic_new.far")
        if use_cache and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode='r')['tokenize_and_classify']
            logging.info(f'ClassifyFst.fst was restored from {os.path.abspath(far_file)}')
        else:
            cardinal = CardinalFst(deterministic=deterministic)
            cardinal_graph = cardinal.fst

            ordinal = OrdinalFst(cardinal=cardinal, deterministic=deterministic)
            ordinal_graph = ordinal.fst

            decimal = DecimalFst(cardinal=cardinal, deterministic=deterministic)
            decimal_graph = decimal.fst
            fraction = FractionFst(deterministic=deterministic, cardinal=cardinal)
            fraction_graph = fraction.fst

            measure = MeasureFst(cardinal=cardinal, decimal=decimal, fraction=fraction, deterministic=deterministic)
            measure_graph = measure.fst
            date_graph = DateFst(cardinal=cardinal, deterministic=deterministic).fst
            word_graph = WordFst(deterministic=deterministic).graph
            time_graph = TimeFst(cardinal=cardinal, deterministic=deterministic).fst
            telephone_graph = TelephoneFst(deterministic=deterministic).fst
            electronic_graph = ElectronicFst(deterministic=deterministic).fst
            money_graph = MoneyFst(cardinal=cardinal, decimal=decimal, deterministic=deterministic).fst
            whitelist_graph = WhiteListFst(input_case=input_case, deterministic=deterministic).graph
            punct_graph = PunctuationFst(deterministic=deterministic).graph

            # VERBALIZER
            from nemo_text_processing.text_normalization.en.verbalizers.cardinal import CardinalFst as vCardinal
            from nemo_text_processing.text_normalization.en.verbalizers.date import DateFst as vDate
            from nemo_text_processing.text_normalization.en.verbalizers.decimal import DecimalFst as vDecimal
            from nemo_text_processing.text_normalization.en.verbalizers.electronic import ElectronicFst as vElectronic
            from nemo_text_processing.text_normalization.en.verbalizers.fraction import FractionFst as vFraction
            from nemo_text_processing.text_normalization.en.verbalizers.measure import MeasureFst as vMeasure
            from nemo_text_processing.text_normalization.en.verbalizers.money import MoneyFst as vMoney
            from nemo_text_processing.text_normalization.en.verbalizers.ordinal import OrdinalFst as vOrdinal
            from nemo_text_processing.text_normalization.en.verbalizers.roman import RomanFst as vRoman
            from nemo_text_processing.text_normalization.en.verbalizers.telephone import TelephoneFst as vTelephone
            from nemo_text_processing.text_normalization.en.verbalizers.time import TimeFst as vTime
            from nemo_text_processing.text_normalization.en.verbalizers.whitelist import WhiteListFst as vWhiteList
            from nemo_text_processing.text_normalization.en.verbalizers.abbreviation import (
                AbbreviationFst as vAbbreviation,
            )

            cardinal = vCardinal(deterministic=deterministic)
            v_cardinal_graph = cardinal.fst
            decimal = vDecimal(cardinal=cardinal, deterministic=deterministic)
            v_decimal_graph = decimal.fst
            ordinal = vOrdinal(deterministic=deterministic)
            v_ordinal_graph = ordinal.fst
            fraction = vFraction(deterministic=deterministic)
            v_fraction_graph = fraction.fst
            v_telephone_graph = vTelephone(deterministic=deterministic).fst
            v_electronic_graph = vElectronic(deterministic=deterministic).fst
            measure = vMeasure(decimal=decimal, cardinal=cardinal, fraction=fraction, deterministic=deterministic)
            v_measure_graph = measure.fst
            v_time_graph = vTime(deterministic=deterministic).fst
            v_date_graph = vDate(ordinal=ordinal, deterministic=deterministic).fst
            v_money_graph = vMoney(decimal=decimal, deterministic=deterministic).fst
            v_whitelist_graph = vWhiteList(deterministic=deterministic).fst
            v_roman_graph = vRoman(deterministic=deterministic).fst
            v_word = WordFst(deterministic=deterministic).fst
            v_abbreviation = vAbbreviation(deterministic=deterministic).fst

            classify_and_verbalize = (
                pynutil.add_weight(whitelist_graph, 1.01)
                | pynutil.add_weight(pynini.compose(time_graph, v_time_graph), 1.1)
                | pynutil.add_weight(pynini.compose(decimal_graph, v_decimal_graph), 1.1)
                | pynutil.add_weight(pynini.compose(measure_graph, v_measure_graph), 1.1)
                | pynutil.add_weight(pynini.compose(cardinal_graph, v_cardinal_graph), 1.1)
                | pynutil.add_weight(pynini.compose(ordinal_graph, v_ordinal_graph), 1.1)
                | pynutil.add_weight(pynini.compose(telephone_graph, v_telephone_graph), 1.1)
                | pynutil.add_weight(pynini.compose(electronic_graph, v_electronic_graph), 1.1)
                | pynutil.add_weight(pynini.compose(fraction_graph, v_fraction_graph), 1.1)
                | pynutil.add_weight(pynini.compose(money_graph, v_money_graph), 1.1)
                | pynutil.add_weight(word_graph, 100)
                | pynutil.add_weight(pynini.compose(date_graph, v_date_graph), 1.09)
            ).optimize()

            if not deterministic:
                roman_graph = RomanFst(deterministic=deterministic).fst
                # the weight matches the word_graph weight for "I" cases in long sentences with multiple semiotic tokens
                classify_and_verbalize |= pynutil.add_weight(pynini.compose(roman_graph, v_roman_graph), 100)

                abbreviation_graph = AbbreviationFst(deterministic=deterministic).fst
                classify_and_verbalize |= pynutil.add_weight(pynini.compose(abbreviation_graph, v_abbreviation), 100)

            # from pynini.lib.rewrite import top_rewrites
            # print([print(x) for x in top_rewrites("$1.01", money_graph, 20)])
            # import pdb;
            # pdb.set_trace()
            # print(top_rewrites("No. 5", classify_and_verbalize, 5))

            punct = pynutil.add_weight(punct_graph, weight=1.1)
            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(" "))
                + classify_and_verbalize
                + pynini.closure(pynutil.insert(" ") + punct)
            )

            graph = token_plus_punct + pynini.closure(delete_extra_space + token_plus_punct)
            graph = delete_space + graph + delete_space

            self.fst = graph.optimize()
            _generator_main(far_file, {"tokenize_and_classify": self.fst})
