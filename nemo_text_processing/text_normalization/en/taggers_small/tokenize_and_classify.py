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

from nemo_text_processing.text_normalization.en import taggers, taggers_small
from nemo_text_processing.text_normalization.en.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)

from nemo.utils import logging

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class ClassifyFstSmall(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence including punctuation.
    For deployment, this grammar will be compiled and exported to OpenFst Finate State Archiv (FAR) File. 
    More details to deployment at NeMo/tools/text_processing_deployment.
    
    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
        whitelist: path to a file with whitelist replacements
    """

    def __init__(
        self,
        input_case: str,
        deterministic: bool = True,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, f"_{input_case}_en_tn_small_{deterministic}_deterministic.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logging.info(f'ClassifyFst.fst was restored from {far_file}.')
        else:
            logging.info(f"Creating ClassifyFst grammars.")
            cardinal_small = taggers_small.CardinalFst(deterministic=deterministic)
            default_cardinal = taggers.CardinalFst(deterministic=deterministic)
            cardinal_graph = cardinal_small.fst

            # ordinal = taggers_small.OrdinalFst(cardinal=cardinal, deterministic=deterministic)
            # ordinal_graph = ordinal.fst

            decimal_small = taggers_small.DecimalFst(cardinal=cardinal_small, deterministic=deterministic)
            decimal_graph = decimal_small.fst
            fraction = taggers_small.FractionFst(deterministic=deterministic, small_cardinal=cardinal_small)
            fraction_graph = fraction.fst

            measure = taggers_small.MeasureFst(
                small_cardinal=cardinal_small,
                small_decimal=decimal_small,
                fraction=taggers.FractionFst(default_cardinal),
                deterministic=deterministic,
            )
            measure_graph = measure.fst
            # date_graph = DateFst(cardinal=cardinal, deterministic=deterministic).fst

            word_graph = taggers.WordFst(deterministic=deterministic).fst
            # time_graph = TimeFst(cardinal=cardinal, deterministic=deterministic).fst
            # telephone_graph = TelephoneFst(deterministic=deterministic).fst
            electronic_graph = taggers_small.ElectronicFst(deterministic=deterministic).fst
            money_graph = taggers_small.MoneyFst(
                small_cardinal=cardinal_small, small_decimal=decimal_small, deterministic=deterministic,
            ).fst
            whitelist_graph = taggers_small.WhiteListFst(input_case=input_case, deterministic=deterministic,).fst
            punct_graph = taggers.PunctuationFst(deterministic=deterministic).fst

            classify = (
                # pynutil.add_weight(whitelist_graph, 1.01)
                # | pynutil.add_weight(decimal_graph, 1.1)
                pynutil.add_weight(measure_graph, 1.1)
                # | pynutil.add_weight(cardinal_graph, 1.1)
                # pynutil.add_weight(money_graph, 1.1)
                # | pynutil.add_weight(electronic_graph, 1.1)
                # pynutil.add_weight(fraction_graph, 1.1)
                | pynutil.add_weight(word_graph, 100)
            )

            punct = pynutil.insert("tokens { ") + pynutil.add_weight(punct_graph, weight=1.1) + pynutil.insert(" }")
            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(" ")) + token + pynini.closure(pynutil.insert(" ") + punct)
            )

            graph = token_plus_punct + pynini.closure(delete_extra_space + token_plus_punct)
            graph = delete_space + graph + delete_space

            self.fst = graph.optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")
