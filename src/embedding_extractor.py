import torch.nn as nn


class EmbeddingExtractor:
    """
        Koppelt sich an ein PyTorch-Modul an, um Zwischen-Embeddings zu extrahieren

        Speichert die Ausgabe einer spezifischen Schicht während des Forward Passes
        Wird im Projekt verwendet, um Features aus den Layer-Ausgaben von YOLO und MiVOLO
        für die Feature-Distillation abzugreifen

        Parameters
        ----------
        model : nn.Module
            Das PyTorch-Modell, aus dem Embeddings extrahiert werden sollen
        layer_name : str
            Der Name der Zielschicht im Modell

        Raises
        ------
        ValueError
            Falls der angegebene `layer_name` im Modell nicht existiert """


    def __init__(self, model: nn.Module, layer_name: str):
        self.model = model
        self.layer_name = layer_name
        self.embedding = None
        self.hook = None
        self._register_hook()

    def _get_layer(self):
        """Sucht das Modul nach Layer Namen im Modell"""
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                return module

        available_layers = [name for name, _ in self.model.named_modules()]
        raise ValueError(f"Layer '{self.layer_name}' wurde nicht gefunden.\n"f"Verfügbare Layer:\n{available_layers}")

    def _hook_fn(self, module, inputs, output):
        """
        Hook-Callback zur Zwischenspeicherung des Ausgabe-Tensors.

        Parameters
        ----------
        module : nn.Module
             Das Modul, an dem der Hook ausgeführt wird
        inputs : tuple
            Eingabetensoren des Layers
        output : torch.Tensor or tuple
            Ausgabetensor des Layers """
        if isinstance(output, (tuple, list)):
            output = output[0]

        self.embedding = output

    def _register_hook(self):
        """Koppelt die Hook-Funktion an den gewählten Layer"""
        layer = self._get_layer()
        self.hook = layer.register_forward_hook(self._hook_fn)
        print(f"Hook registriert auf Layer: {self.layer_name} ({layer.__class__.__name__})")

    def get_embedding(self):
        """
        Gibt das zuletzt gespeicherte Embedding des Forward Passes zurück

        Returns
        -------
        torch.Tensor
            Der extrahierte Feature-Tensor der Zielschicht

        Raises
        ------
        RuntimeError
            Falls aufgerufen wird, bevor ein Forward Pass ausgeführt wurde """
        if self.embedding is None:
            raise RuntimeError(f"Noch kein Embedding für Layer '{self.layer_name}' gespeichert. "f"Erst Forward Pass ausführen.")

        return self.embedding

    def remove(self):
        """Entfernt den Hook vom Modell zur Vermeidung von VRAM-Speicherlecks"""
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

