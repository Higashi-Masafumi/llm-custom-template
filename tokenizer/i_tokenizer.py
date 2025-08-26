from abc import ABC, abstractmethod


class ITokenizer(ABC):
    """Tokenizer interface."""

    @property
    @abstractmethod
    def eos_id(self) -> int:
        """End of sentence ID."""
        pass

    @property
    @abstractmethod
    def pad_id(self) -> int:
        """Padding ID."""
        pass

    @abstractmethod
    def encode(self, s: str) -> list[int]:
        """Encode a string into a list of integers."""
        pass

    @abstractmethod
    def decode(self, t: list[int]) -> str:
        """Decode a list of integers into a string."""
        pass
