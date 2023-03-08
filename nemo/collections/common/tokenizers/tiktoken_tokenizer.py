# Copyright (c) 2022, EleutherAI.  All rights reserved.
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
from typing import List, Union

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ModuleNotFoundError:
    HAS_TIKTOKEN = False

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

__all__ = ['TikTokenTokenizer']


class TikTokenTokenizer(TokenizerSpec):
    """Tokenizer from OpenAI's tiktoken implementation"""

    def __init__(self, encoding_name: str):
        """
        Args:
            encoding_name: name of the encoding.
        """
        assert HAS_TIKTOKEN, "`tiktoken`(https://github.com/openai/tiktoken) is not installed. Please install it with `pip install tiktoken`"
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab

    def text_to_tokens(self, text: str):
        return self.tokenizer.encode_ordinary(text)

    def tokens_to_text(self, tokens: List[int]):
        return self.tokenizer.decode(tokens)

    def tokens_to_ids(self, tokens):
        raise NotImplementedError("To be implemented")

    def _convert_tokens_to_ids(self, tokens: Union[str, List[str]]):
        raise NotImplementedError("To be implemented")

    def ids_to_tokens(self, ids):
        raise NotImplementedError("To be implemented")

    def _convert_ids_tokens_to_string(self, ids: Union[int, List[int]]):
        raise NotImplementedError("To be implemented")

    def text_to_ids(self, text):
        raise NotImplementedError("To be implemented")

    def ids_to_text(self, ids):
        raise NotImplementedError("To be implemented")

    @property
    def add_special_tokens(self, special_tokens_dict: dict) -> int:
        raise NotImplementedError("Extending `tiktoken`'s vocab is not yet supported")

    @property
    def eos_id(self):
        return self.tokenizer.eot_token
    
    @property
    def eod_id(self):
        return self.tokenizer.eod_token

    @property
    def unk_id(self):
        return None

    @property
    def bos_id(self):
        return self.tokenizer.eot_token

    @property
    def name(self):
        return type(self.tokenizer).__name__
