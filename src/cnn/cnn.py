# Imports
import torch
import torch.nn as nn
from torchvision import models
from zennit.composites import LayerMapComposite
from zennit.rules import Gamma as ZGamma


"""
model_names: "resnet50" or "resnet18"
"""
class Custom_model:
    def __init__(self, device, name):
        self.model_name = name
        self.device = device

        if name == "resnet50":
            self.model = models.resnet50(weights="IMAGENET1K_V2")
        elif name == "resnet18":
            self.model = models.resnet18(weights="IMAGENET1K_V1")
        else:
            raise ValueError(f"Unknown model: {name} (use 'resnet50' or 'resnet18')")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Raw capture dicts, keyed by conv layer name (e.g. "layer1.0.conv2")
        self.activations = {}
        self.gradients = {}

        self._register_hooks()

    def get_model(self):
        return self.model

    # ------------------------------------------------------------------
    # Hook registration — one forward hook + one full_backward hook per
    # Conv2d layer, capturing raw activations/gradients (no aggregation,
    # no ReLU filtering — that happens later in the scoring scripts).
    # ------------------------------------------------------------------
    def _save_activation(self, layer_name):
        def hook(module, input, output):
            self.activations[layer_name] = output
        return hook

    def _save_gradient(self, layer_name):
        def hook(module, grad_input, grad_output):
            # grad_output[0] shape: [Batch, Channels, H, W]
            self.gradients[layer_name] = grad_output[0]
        return hook

    def _register_hooks(self):
        for layer_name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(self._save_activation(layer_name))
                module.register_full_backward_hook(self._save_gradient(layer_name))

    def clear(self):
        """Wipes tracking dicts to clean memory between runs."""
        self.activations = {}
        self.gradients = {}

    # ------------------------------------------------------------------
    # Plain capture pass — one forward + one backward, returns raw
    # activations and gradients for every Conv2d layer. All GMAR /
    # GMAR++ / LeGrad / magnitude / Taylor-style scores can be computed
    # downstream from these two dicts without needing a fresh pass per
    # method.
    # ------------------------------------------------------------------
    def forward_pass(self, pixel_values, target_class=None):
        """
        Runs a forward pass and a batch-safe backward pass, capturing
        raw per-channel activations and gradients for every Conv2d
        layer in the network.

        Returns:
            output: model logits, shape [B, num_classes]
            activations: dict {layer_name: tensor [B, C, H, W]}
            gradients: dict {layer_name: tensor [B, C, H, W]}
        """
        self.clear()

        pixel_values = pixel_values.to(self.device)

        output = self.model(pixel_values)

        if target_class is None:
            target_class = output.argmax(dim=1)

        batch_indices = torch.arange(output.size(0), device=output.device)
        target_scores = output[batch_indices, target_class]
        loss_scalar = target_scores.sum()

        self.model.zero_grad()
        loss_scalar.backward()

        return output, self.activations, self.gradients

    # ------------------------------------------------------------------
    # Causal ablation — needs its own forward pass since a channel is
    # zeroed BEFORE running forward, not analyzed via gradients after.
    # ------------------------------------------------------------------
    def _get_channel_ablation_hook(self, channel_idx):
        def hook(module, input, output):
            output = output.clone()
            output[:, channel_idx, :, :] = 0.0
            return output
        return hook

    def causal_forward_pass(self, pixel_values, layer_name=None, channel_idx=None):
        """
        Runs a forward pass, optionally zero-ablating one output channel
        of a given Conv2d layer (by module name from named_modules()).
        layer_name/channel_idx both None -> normal, unmodified forward pass.
        """
        pixel_values = pixel_values.to(self.device)

        handle = None
        if layer_name is not None and channel_idx is not None:
            target_module = dict(self.model.named_modules())[layer_name]
            handle = target_module.register_forward_hook(
                self._get_channel_ablation_hook(channel_idx)
            )
        try:
            output = self.model(pixel_values)
        finally:
            if handle is not None:
                handle.remove()
        return output

    # ------------------------------------------------------------------
    # LRP (Gamma-rule) — needs its own forward pass since the composite
    # must be registered BEFORE backward runs, changing how gradients
    # are computed (Gamma-rule relevance instead of plain gradients).
    # Capture mechanism (activations/gradients dicts) is identical to
    # forward_pass() above -- only the backward math differs.
    # ------------------------------------------------------------------
    def lrp_forward_pass(self, pixel_values, target_class=None, conv_gamma=0.25):
        """
        LRP (Gamma-rule) capture pass: applies zennit's Gamma-rule
        composite over the model's Conv2d layers so that .grad on each
        conv's output carries LRP relevance (redistributed via the
        Gamma rule) rather than a plain autograd gradient -- same
        principle as vit.py's lrp_forward_pass, but scoped to
        convolutions instead of Linear layers (matches the paper's
        Conv. column in Table A.2, gamma=0.25).

        Returns the same (output, activations, gradients) shape as
        forward_pass(), so downstream scoring scripts don't need to
        know which backward rule produced the gradients.
        """
        self.clear()

        pixel_values = pixel_values.to(self.device)

        composite = LayerMapComposite([
            (nn.Conv2d, ZGamma(conv_gamma)),
        ])
        composite.register(self.model)

        try:
            output = self.model(pixel_values)

            if target_class is None:
                target_class = output.argmax(dim=1)

            batch_indices = torch.arange(output.size(0), device=output.device)
            target_scores = output[batch_indices, target_class]
            loss_scalar = target_scores.sum()

            self.model.zero_grad()
            loss_scalar.backward()
        finally:
            # Always remove the composite, even if backward fails, so it
            # doesn't leak into a later plain forward_pass() call.
            composite.remove()

        return output, self.activations, self.gradients