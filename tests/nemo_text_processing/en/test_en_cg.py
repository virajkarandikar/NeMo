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

import pytest
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio
from parameterized import parameterized

from ..utils import CACHE_DIR, PYNINI_AVAILABLE, parse_test_case_file


class TestDate:
    normalizer_en = (
        Normalizer(input_case='cased', lang='en_small', cache_dir=CACHE_DIR, overwrite_cache=False)
        if PYNINI_AVAILABLE
        else None
    )

    @parameterized.expand(parse_test_case_file('en/data_cg_text_normalization/test_cases_en_cg_fraction.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE, reason="`pynini` not installed, please install via nemo_text_processing/setup.sh"
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm_fraction(self, test_input, expected):
        pred = self.normalizer_en.normalize(test_input, verbose=False)
        assert pred == expected, f"For input: '{test_input}': {pred} != {expected}"

    @parameterized.expand(parse_test_case_file('en/data_cg_text_normalization/test_cases_en_cg_cardinal.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE, reason="`pynini` not installed, please install via nemo_text_processing/setup.sh"
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm_cardinal(self, test_input, expected):
        pred = self.normalizer_en.normalize(test_input, verbose=False)
        assert pred == expected, f"For input: '{test_input}': {pred} != {expected}"

    @parameterized.expand(parse_test_case_file('en/data_cg_text_normalization/test_cases_en_cg_decimal.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE, reason="`pynini` not installed, please install via nemo_text_processing/setup.sh"
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm_decimal(self, test_input, expected):
        pred = self.normalizer_en.normalize(test_input, verbose=False)
        assert pred == expected, f"For input: '{test_input}': {pred} != {expected}"

    @parameterized.expand(parse_test_case_file('en/data_cg_text_normalization/test_cases_en_cg_money.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE, reason="`pynini` not installed, please install via nemo_text_processing/setup.sh"
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm_money(self, test_input, expected):
        pred = self.normalizer_en.normalize(test_input, verbose=False)
        assert pred == expected, f"For input: '{test_input}': {pred} != {expected}"
