import torch.nn as nn
import torch.nn.functional as F

class EmbeddingExtractor:

    def __init__(self, model: nn.Module, layer_name: str):
        self.model = model
        self.layer_name = layer_name
        self.embedding = None
        self.hook = None
        self._register_hook()

    def _get_layer(self):
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                return module

        available_layers = [name for name, _ in self.model.named_modules()]
        raise ValueError(f"Layer '{self.layer_name}' wurde nicht gefunden.\n"f"Verfügbare Layer:\n{available_layers}")

    def _hook_fn(self, module, inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]

        self.embedding = output

    def _register_hook(self):
        layer = self._get_layer()
        self.hook = layer.register_forward_hook(self._hook_fn)
        print(f"Hook registriert auf Layer: {self.layer_name} ({layer.__class__.__name__})")

    def get_embedding(self):
        if self.embedding is None:
            raise RuntimeError(f"Noch kein Embedding für Layer '{self.layer_name}' gespeichert. "f"Erst Forward Pass ausführen.")

        return self.embedding

    def remove(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

