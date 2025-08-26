import json
from pathlib import Path
from typing import Final

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from model_config import ModelConfig
from models import BaseModel
from tokenizer import ITokenizer


class ModelLoader:
    """Load a model from a model folder.

    Args:
        model_folder (str): The path to the model folder.
    """

    CONFIG_FILE: Final[str] = "config.json"
    TOKENIZER_FILE: Final[str] = "tokenizer.model"
    WEIGHTS_FILE: Final[str] = "weights.npz"

    def __init__(self, model_folder: str):
        self._model_path = Path(model_folder)
        self._config_path = self._model_path / self.CONFIG_FILE
        self._tokenizer_path = self._model_path / self.TOKENIZER_FILE

    def load_model(
        self,
        model_class: type[BaseModel],
        tokenizer_class: type[ITokenizer],
    ) -> tuple[BaseModel, ITokenizer]:
        """Load the model from the model folder.

        Args:
            model (BaseModel): The model to load.
            tokenizer (SentencePieceTokenizer): The tokenizer to load.

        Returns:
            tuple[BaseModel, SentencePieceTokenizer]: The model and the tokenizer.
        """
        # Load tokenizer
        tokenizer = tokenizer_class(self._tokenizer_path)

        # Load model config
        with open(self._config_path, "r") as f:
            config = json.loads(f.read())
            config.pop("sliding_window", None)
            config.pop("model_type", None)
            quantization = config.pop("quantization", None)
            model_config = ModelConfig(**config)

        # Load weights
        weights = mx.load(self._model_path / self.WEIGHTS_FILE)
        weights = tree_unflatten(list(weights.items()))
        model = model_class(config=model_config)

        # Quantize model
        if quantization is not None:
            nn.quantize(model=model, **quantization)

        # Update model weights
        model.update(weights)

        # Evaluate model
        mx.eval(model.parameters())
        return model, tokenizer
