import os
from typing import List

import sentencepiece as spm
import logging

LOG = logging.getLogger()


class Dictionary:
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class LettersDict(Dictionary):
    def __init__(self, dict_path: str):
        with open(dict_path) as f:
            self._letters = [letter.replace("\n", "") for letter in f.readlines()]
        self._ids_to_letters = dict((i, letter) for i, letter in enumerate(self._letters))
        self._letters_to_ids = dict((letter, i) for i, letter in enumerate(self._letters))

        self._validate_vocabulary()

    def encode(self, text: str) -> List[int]:
        ids = [self._letters_to_ids.get(x) for x in text]
        return [i for i in ids if i is not None]

    def decode(self, ids: List[int]) -> str:
        return "".join(self._ids_to_letters.get(i, "") for i in ids)

    def blank_id(self) -> int:
        return len(self._letters) - 1

    def _validate_vocabulary(self):
        if self._letters[-1] != "|":
            raise Exception("last letter expected to be blank, got {}".format(self._letters[-1]))
        for letter in self._letters:
            if letter == " ":
                return
        raise Exception("no space in dictionary")

    def __len__(self):
        return len(self._letters)


class SentencePieceDict(Dictionary):
    def __init__(self, dict_path: str):
        with open(dict_path, "rb") as f:
            self._serialized_dict = f.read()
        self._sp_model = spm.SentencePieceProcessor()
        self._sp_model.load_from_serialized_proto(self._serialized_dict)
        self._dict_file = os.path.basename(dict_path)

    def encode(self, text: str) -> List[int]:
        ids = self._sp_model.encode_as_ids(text) if len(text) > 0 else [self.eos_id()]
        return ids

    def decode(self, ids: List[int]) -> str:
        # ignore all ids after eos_id
        n = next((i for i, x in enumerate(ids) if x == self.eos_id()), len(ids))
        return self._sp_model.decode_ids(ids[:n])

    def unk_id(self) -> int:
        return self._sp_model.unk_id()

    def pad_id(self) -> int:
        id = self._sp_model.pad_id()
        assert id != self.unk_id(), "'pad' token is not defined"
        return id

    def bos_id(self) -> int:
        id = self._sp_model.bos_id()
        assert id != self.unk_id(), "'bos' token is not defined"
        return id

    def eos_id(self) -> int:
        id = self._sp_model.eos_id()
        assert id != self.unk_id(), "'eos' token is not defined"
        return id

    def sil_id(self) -> int:
        id = self._sp_model.piece_to_id("<SIL>")
        assert id != self.unk_id(), "'sil' token is not defined"
        return id

    def __len__(self):
        return len(self._sp_model)

    def __getstate__(self):
        return {"dict_file": self._dict_file, "serialized_dict": self._serialized_dict}

    def __setstate__(self, state):
        self._dict_file = state["dict_file"]
        self._serialized_dict = state["serialized_dict"]
        self._sp_model = spm.SentencePieceProcessor()
        self._sp_model.load_from_serialized_proto(self._serialized_dict)
