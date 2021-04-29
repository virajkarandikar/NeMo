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

import itertools
import os
from argparse import ArgumentParser
from collections import OrderedDict
from typing import List

from nemo_text_processing.text_normalization.taggers.tokenize_and_classify import ClassifyFst
from nemo_text_processing.text_normalization.token_parser import PRESERVE_ORDER_KEY, TokenParser
from nemo_text_processing.text_normalization.verbalizers.verbalize_final import VerbalizeFinalFst
from tqdm import tqdm

from nemo.collections import asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate

try:
    import pynini

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class Normalizer:
    """
    Normalizer class that converts text from written to spoken form. 
    Useful for TTS preprocessing. 

    Args:
        input_case: expected input capitalization
    """

    def __init__(self, input_case: str):
        assert input_case in ["lower_cased", "cased"]

        self.tagger = ClassifyFst(input_case=input_case)
        self.verbalizer = VerbalizeFinalFst()
        self.parser = TokenParser()

    def normalize_list(self, texts: List[str], verbose=False) -> List[str]:
        """
        NeMo text normalizer 

        Args:
            texts: list of input strings
            verbose: whether to print intermediate meta information

        Returns converted list input strings
        """
        res = []
        for input in tqdm(texts):
            try:
                text = self.normalize(input, verbose=verbose)
            except:
                print(input)
                raise Exception
            res.append(text)
        return res

    def normalize(self, text: str, verbose: bool) -> str:
        """
        Main function. Normalizes tokens from written to spoken form
            e.g. 12 kg -> twelve kilograms

        Args:
            text: string that may include semiotic classes
            verbose: whether to print intermediate meta information

        Returns: spoken form
        """
        text = text.strip()
        if not text:
            if verbose:
                print(text)
            return text
        text = pynini.escape(text)
        tagged_lattice = self.find_tags(text)
        tagged_text = self.select_tag(tagged_lattice)
        if verbose:
            print(tagged_text)
        self.parser(tagged_text)
        tokens = self.parser.parse()
        tags_reordered = self.generate_permutations(tokens)
        for tagged_text in tags_reordered:
            tagged_text = pynini.escape(tagged_text)
            verbalizer_lattice = self.find_verbalizer(tagged_text)
            if verbalizer_lattice.num_states() == 0:
                continue
            output = self.select_verbalizer(verbalizer_lattice)
            return output
        raise ValueError()

    def normalize_with_audio(self, text: str, verbose: bool) -> str:
        """
        Main function. Normalizes tokens from written to spoken form
            e.g. 12 kg -> twelve kilograms

        Args:
            text: string that may include semiotic classes
            verbose: whether to print intermediate meta information

        Returns: spoken form
        """
        text = text.strip()
        if not text:
            if verbose:
                print(text)
            return text
        text = pynini.escape(text)
        tagged_lattice = self.find_tags(text)
        tagged_texts = self.select_all_semiotic_tags(tagged_lattice)
        normalized_texts = []
        for tagged_text in tagged_texts:
            self.parser(tagged_text)
            tokens = self.parser.parse()
            tags_reordered = self.generate_permutations(tokens)
            for tagged_text in tags_reordered:
                tagged_text = pynini.escape(tagged_text)
                verbalizer_lattice = self.find_verbalizer(tagged_text)
                if verbalizer_lattice.num_states() == 0:
                    continue
                normalized_texts.append(self.select_verbalizer(verbalizer_lattice))
        return normalized_texts

    def _permute(self, d: OrderedDict) -> List[str]:
        """
        Creates reorderings of dictionary elements and serializes as strings

        Args:
            d: (nested) dictionary of key value pairs

        Return permutations of different string serializations of key value pairs
        """
        l = []
        if PRESERVE_ORDER_KEY in d.keys():
            d_permutations = [d.items()]
        else:
            d_permutations = itertools.permutations(d.items())
        for perm in d_permutations:
            subl = [""]
            for k, v in perm:
                if isinstance(v, str):
                    subl = ["".join(x) for x in itertools.product(subl, [f"{k}: \"{v}\" "])]
                elif isinstance(v, OrderedDict):
                    rec = self._permute(v)
                    subl = ["".join(x) for x in itertools.product(subl, [f" {k} {{ "], rec, [f" }} "])]
                elif isinstance(v, bool):
                    subl = ["".join(x) for x in itertools.product(subl, [f"{k}: true "])]
                else:
                    raise ValueError()
            l.extend(subl)
        return l

    def generate_permutations(self, tokens: List[dict]):
        """
        Generates permutations of string serializations of list of dictionaries

        Args:
            tokens: list of dictionaries

        Returns string serialization of list of dictionaries
        """

        def _helper(prefix: str, tokens: List[dict], idx: int):
            """
            Generates permutations of string serializations of given dictionary

            Args:
                tokens: list of dictionaries
                prefix: prefix string
                idx:    index of next dictionary

            Returns string serialization of dictionary
            """
            if idx == len(tokens):
                yield prefix
                return
            token_options = self._permute(tokens[idx])
            for token_option in token_options:
                yield from _helper(prefix + token_option, tokens, idx + 1)

        return _helper("", tokens, 0)

    def find_tags(self, text: str) -> 'pynini.FstLike':
        """
        Given text use tagger Fst to tag text

        Args:
            text: sentence

        Returns: tagged lattice
        """
        lattice = text @ self.tagger.fst
        return lattice

    def select_all_semiotic_tags(self, lattice: 'pynini.FstLike') -> List[str]:
        all_semiotic_tags = []
        all = lattice.paths("utf8")

        semiotic_classes = ['date', 'cardinal', 'ordinal', 'decimal', 'money', 'time', 'telephone', 'electronic']
        for item in all.items():
            for semiotic in semiotic_classes:
                if semiotic in item[1]:
                    all_semiotic_tags.append(item[1])
        return all_semiotic_tags

    def select_tag(self, lattice: 'pynini.FstLike') -> str:
        """
        Given tagged lattice return shortest path

        Args:
            tagged_text: tagged text

        Returns: shortest path
        """
        tagged_text = pynini.shortestpath(lattice, nshortest=1, unique=True).string()
        return tagged_text

    def find_verbalizer(self, tagged_text: str) -> 'pynini.FstLike':
        """
        Given tagged text creates verbalization lattice
        This is context-independent.

        Args:
            tagged_text: input text

        Returns: verbalized lattice
        """
        lattice = tagged_text @ self.verbalizer.fst
        return lattice

    def select_verbalizer(self, lattice: 'pynini.FstLike') -> str:
        """
        Given verbalized lattice return shortest path

        Args:
            lattice: verbalization lattice

        Returns: shortest path
        """
        output = pynini.shortestpath(lattice, nshortest=1, unique=True).string()
        return output


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_string",
        help="input string",
        type=str,
        default="and after a long interval his dead body was discovered, shockingly disfigured, in a ditch. This was in 1802.",
    )
    parser.add_argument(
        "--input_case", help="input capitalization", choices=["lower_cased", "cased"], default="lower_cased", type=str
    )
    parser.add_argument("--verbose", help="print info for debugging", action='store_true')
    parser.add_argument(
        '--model', type=str, default='QuartzNet15x5Base-En', help='Pre-trained model name or path to model checkpoint'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    normalizer = Normalizer(input_case=args.input_case)
    audio = '/mnt/sdb/DATA/normalization/wavs_16000/lj_speech/LJ008-0149.wav'
    model = 'QuartzNet15x5Base-En'

    if args.model is None:
        print(f"No model provided, vocabulary won't be used")
    elif os.path.exists(args.model):
        asr_model = nemo_asr.models.EncDecCTCModel.restore_from(args.model)
        vocabulary = asr_model.cfg.decoder.vocabulary
    elif args.model in nemo_asr.models.EncDecCTCModel.get_available_model_names():
        asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(args.model)
        vocabulary = asr_model.cfg.decoder.vocabulary
    else:
        raise ValueError(
            f'Provide path to the pretrained checkpoint or choose from {nemo_asr.models.EncDecCTCModel.get_available_model_names()}'
        )
    transcript = asr_model.transcribe([audio])[0]

    print(f'input     : {args.input_string}')
    print(f'transcript: {transcript}')
    print('=' * 20)

    normalized_texts = normalizer.normalize_with_audio(args.input_string, verbose=args.verbose)

    if len(normalized_texts) == 0:
        raise ValueError()

    for i in range(len(normalized_texts)):
        normalized_texts[i] = normalized_texts[i].replace(' ,', ',').replace(' .', '.')
    normalized_texts = set(normalized_texts)

    print('normalized:')
    for text in normalized_texts:
        wer = round(word_error_rate([transcript], [text], use_cer=True) * 100, 2)
        print(text, wer)
