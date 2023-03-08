# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from nemo.collections.common.tokenizers.tiktoken_tokenizer import TikTokenTokenizer


class TestTikTokenTokenizer:
    name = "cl100k_base"

    @pytest.mark.unit
    def test_text_to_tokens(self, test_data_dir):
        tokenizer = TikTokenTokenizer(self.name)

        text = "<|endoftext|> a b c e <|endoftext|> f g h i <|endoftext|>"
        tokens = tokenizer.text_to_tokens(text)

        assert tokens.count("<|endoftext|>") == 0

    @pytest.mark.unit
    def test_tokens_to_text(self, test_data_dir):
        tokenizer = TikTokenTokenizer(self.name)

        text = "a b c e f g h i"
        tokens = tokenizer.text_to_tokens(text)
        result = tokenizer.tokens_to_text(tokens)

        assert text == result

    # @pytest.mark.unit
    # def test_text_to_ids(self):
    #     tokenizer = TikTokenTokenizer(self.name)

    #     text = "<BOS> a b c <UNK> e f g h i <EOS>"
    #     tokens = tokenizer.text_to_ids(text)

    #     assert tokens.count(tokenizer.bos_id) == 0
    #     assert tokens.count(tokenizer.unk_id) == 0
    #     assert tokens.count(tokenizer.eos_id) == 0

    # @pytest.mark.unit
    # def test_ids_to_text(self, test_data_dir):
    #     tokenizer = TikTokenTokenizer(test_data_dir + self.name)

    #     text = "a b c e f g h i"
    #     ids = tokenizer.text_to_ids(text)
    #     result = tokenizer.ids_to_text(ids)

    #     assert text == result

    # @pytest.mark.unit
    # def test_tokens_to_ids(self, test_data_dir):
    #     tokenizer = TikTokenTokenizer(test_data_dir + self.name)

    #     tokens = ["<BOS>", "a", "b", "c", "<UNK>", "e", "f", "<UNK>", "g", "h", "i", "<EOS>"]
    #     ids = tokenizer.tokens_to_ids(tokens)

    #     assert len(ids) == len(tokens)
    #     assert ids.count(tokenizer.bos_id) == 1
    #     assert ids.count(tokenizer.eos_id) == 1
    #     assert ids.count(tokenizer.unk_id) == 2

    # @pytest.mark.unit
    # def test_ids_to_tokens(self):
    #     tokenizer = TikTokenTokenizer(self.name)

    #     tokens = ["<BOS>", "a", "b", "c", "<UNK>", "e", "f", "<UNK>", "g", "h", "i", "<EOS>"]
    #     ids = tokenizer.tokens_to_ids(tokens)
    #     result = tokenizer.ids_to_tokens(ids)

    #     assert len(result) == len(tokens)

    #     for i in range(len(result)):
    #         assert result[i] == tokens[i]
