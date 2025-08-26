from pathlib import Path

from sentencepiece import SentencePieceProcessor


class Tokenizer:
    """Tokenizer class using sentencepiece.(https://arxiv.org/pdf/1808.06226)

    Args:
        model_path (str): The path to the model file.

    Note: https://huggingface.co/docs/transformers/ja/tokenizer_summary
    """

    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_path)
        self._sep = "_"
        assert self._model.vocab_size() == self._model.GetPieceSize()

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        return self._model.pad_id()

    def encode(self, s: str) -> list[int]:
        """Encode a string into a list of integers.

        Args:
            s (str): The string to encode.

        Returns:
            list[int]: The encoded string.
        """
        return self._model.EncodeAsIds(s)

    def decode(self, t: list[int]) -> str:
        """Decode a list of integers into a string.

        Args:
            t (list[int]): The list of integers to decode.

        Returns:
            str: The decoded string.
        """
        return self._model.DecodeIds(t)
