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
    NEMO_LOWER,
    TO_LOWER,
    GraphFst,
    get_abs_path,
    insert_space,
)

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic: as URLs, email addresses, etc.
        e.g. cdf1@abc.edu -> tokens { electronic { username: "cdf1" domain: "abc.edu" } }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        def get_input_symbols(f):
            accepted_symbols = []
            with open(f, 'r') as f:
                for line in f:
                    symbol, _ = line.split('\t')
                    accepted_symbols.append(pynini.accep(symbol))
            return accepted_symbols

        accepted_symbols = get_input_symbols(get_abs_path("data/electronic/symbols.tsv"))
        accepted_common_domains = get_input_symbols(get_abs_path("data/electronic/domain.tsv"))
        accepted_symbols = NEMO_ALPHA + pynini.closure(NEMO_ALPHA | NEMO_DIGIT | pynini.union(*accepted_symbols))
        graph_symbols = pynini.string_file(get_abs_path("data/electronic/symbols.tsv")).optimize()

        username = pynutil.insert("username: \"") + accepted_symbols + pynutil.insert("\"") + pynini.cross('@', ' ')
        domain_graph = accepted_symbols + pynini.accep('.') + accepted_symbols
        domain_graph = pynutil.insert("domain: \"") + domain_graph + pynutil.insert("\"")
        domain_common_graph = (
            pynutil.insert("domain: \"")
            + accepted_symbols
            + pynini.union(*accepted_common_domains)
            + pynutil.insert("\"")
        )

        protocol_start = pynini.accep("https://") | pynini.accep("http://")
        protocol_symbols = pynini.closure(
            (NEMO_ALPHA | pynutil.add_weight(graph_symbols | pynini.cross(":", "colon"), -0.1)) + pynutil.insert(" ")
        )
        protocol_end = pynini.accep("www.")
        protocol = protocol_start | protocol_end | (protocol_start + protocol_end)
        protocol = pynini.compose(protocol, protocol_symbols)
        protocol = pynutil.insert("protocol: \"") + protocol + pynutil.insert("\"")
        graph = username + domain_graph
        graph |= domain_common_graph
        graph |= pynini.closure(protocol, 0, 1) + pynutil.insert(" ") + domain_graph
        graph = graph.optimize()

        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        single_digits_graph = pynini.invert(graph_digit | graph_zero)
        single_digits_graph = single_digits_graph + pynini.closure(insert_space + single_digits_graph)

        # 2.7.7.68 -> two dot seven seven six eight
        dot = pynini.cross(".", " dot ")
        one_two_digits = pynini.compose(NEMO_DIGIT ** (1, 2), single_digits_graph)
        id = one_two_digits + dot + one_two_digits + dot + one_two_digits + dot + one_two_digits
        id = pynutil.insert("protocol: \"") + id + pynutil.insert("\"")

        final_graph = self.add_tokens(graph | id.optimize())
        self.fst = final_graph.optimize()
