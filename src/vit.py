# Imports
import torch
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification

# Main class
"""
model_names: 
1. "WinKawaks/vit-tiny-patch16-224"
2. "WinKawaks/vit-small-patch16-224"
"""
class Custom_model(torch.nn.Module):
    def __init__(self, device, name):
        super().__init__()
        self.model_name = name
        self.device = device
        
        # Load model in eager mode to ensure the internal 4D matrix graph is built
        self.model = ViTForImageClassification.from_pretrained(
            self.model_name,
            attn_implementation="eager"
        )
        self.model = self.model.to(self.device)

        # Explicitly instruct the model to output the 4D attention weights
        # self.model.config.output_attentions = False
        # self.model.config.return_dict = False

        # Lists/Dicts to store our structural data
        self.attentions = []
        self.attention_gradients = {}  # FIX: Initialized as a dictionary

        self.model.eval()

        try:
            self.processor = ViTImageProcessor.from_pretrained(self.model_name, local_files_only=True)
        except Exception as e:
            print("[INFO] Failed to load the processor:", e)
            print("[IMPORTANT] please load the processor separately.")
            self.processor = None
        
    def get_model(self):
        return self.model
    
    def config(self):
        return self.model.config
    
    def _create_tensor_hook(self):
        """Standard hook for full backward passes."""
        def hook(grad):
            self.attention_gradients_list.append(grad.detach().cpu())
        return hook

    def _create_tensor_hook_GMAR(self, layer_idx):
        """GMAR hook: L1 norm of raw gradients (no ReLU) for head importance scores."""
        def hook(grad):
            # grad shape: [Batch, Heads, Tokens, Tokens]
            head_scores = grad.abs().sum(dim=[-2, -1])
            mean_head_scores = head_scores.mean(dim=0)
            self.attention_gradients[layer_idx] = mean_head_scores.detach().cpu().numpy()
            return grad
        return hook

    def _create_tensor_hook_GMARpp(self, layer_idx):
        """GMAR++ hook: ReLU filter gradients, then L2 norm for head importance scores."""
        def hook(grad):
            # grad shape: [Batch, Heads, Tokens, Tokens]
            
            # Step 1 - ReLU filter: keep only positive gradients
            positive_grad = torch.clamp(grad, min=0)
            
            # Step 2 - L2 norm on filtered gradients -> [Batch, Heads]
            head_scores = positive_grad.norm(p=2, dim=[-2, -1])
            
            # OR L1 norm:
            # head_scores = positive_grad.abs().sum(dim=[-2, -1])
            
            # Step 3 - Average across batch -> [Heads]
            mean_head_scores = head_scores.mean(dim=0)
            
            self.attention_gradients[layer_idx] = mean_head_scores.detach().cpu().numpy()
            
            return grad
        return hook

    def _create_tensor_hook_legrad(self, layer_idx):
        """LeGrad hook: ReLU filter gradients, then sum for head importance scores."""
        def hook(grad):
            # grad shape: [Batch, Heads, Tokens, Tokens]
            
            # Step 1 - ReLU filter: keep only positive gradients
            positive_grad = torch.clamp(grad, min=0)
            
            # Step 2 - Sum on filtered gradients -> [Batch, Heads]
            head_scores = positive_grad.sum(dim=[-2, -1])
            
            # Step 3 - Average across batch -> [Heads]
            mean_head_scores = head_scores.mean(dim=0)
            
            self.attention_gradients[layer_idx] = mean_head_scores.detach().cpu().numpy()
            
            return grad
        return hook

    def clear(self):
        """Wipes tracking lists to clean memory between runs."""
        self.attentions = []
        self.attention_gradients = {}  # FIX: Keep as a clean dictionary
        self.attention_gradients_list = []
    
    def full_forward_pass(self, input_tensor, target_class=None):
        """Runs a forward pass and a batch-safe backward pass."""
        self.clear() 
        # Make sure this list is reset every run
        self.attention_gradients_list = []
        self.attentions = []  # Ensure this is reset too
        
        # Forward pass - keep output_attentions=True
        output = self.model(input_tensor, output_attentions=True)
        logits = output.logits
        native_attentions = output.attentions
        
        # Store tensors and register hooks on the active GPU tensors
        for attn_tensor in native_attentions:
            self.attentions.append(attn_tensor)  # Keep on GPU for element-wise math later
            attn_tensor.register_hook(self._create_tensor_hook())
            
        if target_class is None:
            target_class = logits.argmax(dim=-1) 
            
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        target_scores = logits[batch_indices, target_class] 
        loss_scalar = target_scores.sum()
        
        # Clear old gradients and backpropagate
        self.model.zero_grad()
        loss_scalar.backward()
        
        # PyTorch hooks append in forward order, so reversing gives you Layer 1 -> Layer 24
        self.attention_gradients_list.reverse()
        
        return output, self.attentions, self.attention_gradients_list
        
    def gmar_forward_pass(self, inputs, target_class=None):
        """GMAR implementation tracking layer-wise head importance scores."""
        self.clear()

        output = self.model(inputs, output_attentions=True)
        logits = output.logits
        native_attentions = output.attentions

        for layer_idx, attn_tensor in enumerate(native_attentions):
            self.attentions.append(attn_tensor.detach().cpu())

            attn_tensor.retain_grad()
            attn_tensor.register_hook(self._create_tensor_hook_GMAR(layer_idx))

        if target_class is None:
            target_class = logits.argmax(dim=-1)

        batch_indices = torch.arange(logits.size(0), device=logits.device)
        target_scores = logits[batch_indices, target_class]
        loss_scalar = target_scores.sum()

        self.model.zero_grad()
        loss_scalar.backward()

        return output, self.attentions, self.attention_gradients

    def gmarpp_forward_pass(self, inputs, target_class=None):
        """GMAR++ implementation tracking layer-wise head importance scores."""
        self.clear()

        output = self.model(inputs, output_attentions=True)
        logits = output.logits
        native_attentions = output.attentions

        for layer_idx, attn_tensor in enumerate(native_attentions):
            self.attentions.append(attn_tensor.detach().cpu())

            attn_tensor.retain_grad()
            attn_tensor.register_hook(self._create_tensor_hook_GMARpp(layer_idx))

        if target_class is None:
            target_class = logits.argmax(dim=-1)

        batch_indices = torch.arange(logits.size(0), device=logits.device)
        target_scores = logits[batch_indices, target_class]
        loss_scalar = target_scores.sum()

        self.model.zero_grad()
        loss_scalar.backward()

        return output, self.attentions, self.attention_gradients


    def legrad_forward_pass(self, inputs, target_class=None):
        """LeGrad implementation tracking layer-wise head importance scores."""
        self.clear()

        output = self.model(inputs, output_attentions=True)
        logits = output.logits
        native_attentions = output.attentions

        for layer_idx, attn_tensor in enumerate(native_attentions):
            self.attentions.append(attn_tensor.detach().cpu())

            attn_tensor.retain_grad()
            attn_tensor.register_hook(self._create_tensor_hook_legrad(layer_idx))

        if target_class is None:
            target_class = logits.argmax(dim=-1)

        batch_indices = torch.arange(logits.size(0), device=logits.device)
        target_scores = logits[batch_indices, target_class]
        loss_scalar = target_scores.sum()

        self.model.zero_grad()
        loss_scalar.backward()

        return output, self.attentions, self.attention_gradients

    def _create_tensor_hook_chefer(self, layer_idx):
        """Chefer hook: stores raw gradients for later combination with LRP relevance."""
        def hook(grad):
            # grad shape: [Batch, Heads, Tokens, Tokens]
            self.attention_gradients[layer_idx] = grad.detach().cpu()
            return grad
        return hook

    def chefer_forward_pass(self, inputs, target_class=None):
        """Chefer et al. forward pass: stores attention maps and gradients."""
        self.clear()

        output = self.model(inputs, output_attentions=True)
        logits = output.logits
        native_attentions = output.attentions

        for layer_idx, attn_tensor in enumerate(native_attentions):
            self.attentions.append(attn_tensor)  # keep on GPU, needed for LRP

            attn_tensor.retain_grad()
            attn_tensor.register_hook(self._create_tensor_hook_chefer(layer_idx))

        if target_class is None:
            target_class = logits.argmax(dim=-1)

        batch_indices = torch.arange(logits.size(0), device=logits.device)
        target_scores = logits[batch_indices, target_class]
        loss_scalar = target_scores.sum()

        self.model.zero_grad()
        loss_scalar.backward()

        return output, self.attentions, self.attention_gradients




# for getting the causal score
    def _get_head_ablation_hook(self, head_idx, head_dim):
        def hook(module, args):
            x = args[0].clone()
            start, end = head_idx * head_dim, (head_idx + 1) * head_dim
            x[..., start:end] = 0.0
            return (x,)
        return hook

    def causal_forward_pass(self, pixel_values, layer_idx=None, head_idx=None):
        """
        Runs a forward pass, optionally zero-ablating one attention head by
        hooking the input of that layer's o_proj (output/merge projection).
        layer_idx/head_idx both None -> normal, unmodified forward pass.
        """
        handle = None
        if layer_idx is not None and head_idx is not None:
            attn = self.model.vit.layers[layer_idx].attention
            head_dim = attn.o_proj.in_features // self.model.config.num_attention_heads
            handle = attn.o_proj.register_forward_pre_hook(
                self._get_head_ablation_hook(head_idx, head_dim)
            )
        try:
            output = self.model(pixel_values)
        finally:
            if handle is not None:
                handle.remove()
        return output.logits


    
    def _create_tensor_hook_LRP(self, layer_idx):
        """
        LRP hook: captures Gamma-rule relevance flowing into each attention
        tensor for head importance scores.
 
        Under a registered zennit Gamma-rule composite, the value arriving
        here via .grad is LRP relevance (redistributed via the Gamma rule),
        not a plain autograd gradient -- same principle as GMAR's hook, but
        the upstream computation is LRP instead of vanilla backprop.
        """
        def hook(grad):
            # grad shape: [Batch, Heads, Tokens, Tokens]
            head_scores = grad.abs().sum(dim=[-2, -1])
            mean_head_scores = head_scores.mean(dim=0)
            self.attention_gradients[layer_idx] = mean_head_scores.detach().cpu().numpy()
            return grad
        return hook
 
 
# --- Add this forward-pass method inside the Custom_model class,
#     next to your other *_forward_pass methods ---
 
    def lrp_forward_pass(self, inputs, target_class=None, linear_gamma=0.05):
        """
        LRP (Gamma-rule) implementation tracking layer-wise head importance scores.
 
        Uses zennit's Gamma-rule composite over the model's nn.Linear layers
        to redistribute relevance during the backward pass, instead of using
        a plain gradient the way GMAR/GMAR++/LeGrad do.
 
        NOTE: this applies the Gamma rule to Linear layers only (Q/K/V
        projections, output projection, MLP) -- the same scope zennit's own
        Vision Transformer example uses. It does NOT yet apply an
        attention-aware rule to the softmax/matmul inside attention itself
        (that is what "AttnLRP" specifically adds on top of vanilla LRP).
        If your GMAR/LeGrad/Chefer comparisons need exact AttnLRP parity,
        the softmax/matmul operations inside the attention forward would
        need to be patched too -- let me know and I'll write that patch
        for HF's modeling_vit.py attention module specifically.
        """
        self.clear()
 
        # Register the Gamma-rule composite on all Linear layers.
        # (Registering/removing per-call keeps this consistent with your
        # other methods, which don't assume any persistent model state.)
        composite = LayerMapComposite([
            (torch.nn.Linear, z_rules.Gamma(linear_gamma)),
        ])
        composite.register(self.model)
 
        try:
            output = self.model(inputs, output_attentions=True)
            logits = output.logits
            native_attentions = output.attentions
 
            for layer_idx, attn_tensor in enumerate(native_attentions):
                self.attentions.append(attn_tensor.detach().cpu())
 
                attn_tensor.retain_grad()
                attn_tensor.register_hook(self._create_tensor_hook_LRP(layer_idx))
 
            if target_class is None:
                target_class = logits.argmax(dim=-1)
 
            batch_indices = torch.arange(logits.size(0), device=logits.device)
            target_scores = logits[batch_indices, target_class]
            loss_scalar = target_scores.sum()
 
            self.model.zero_grad()
            loss_scalar.backward()
        finally:
            # Always remove the composite, even if the backward pass fails,
            # so it doesn't leak into your GMAR/LeGrad/Chefer runs afterward.
            composite.remove()
 
        return output, self.attentions, self.attention_gradients

    # ------------------------------------------------------------------
    # AttnLRP (lxt "efficient" / Input*Gradient mode)
    # ------------------------------------------------------------------
    #
    # IMPORTANT — unlike lrp_forward_pass (zennit Gamma composite, which
    # is registered and removed around each call), lxt's monkey_patch
    # permanently rewrites the softmax/matmul functions inside
    # modeling_vit for the whole Python process. There is no clean
    # "unpatch" provided by the library. Practical consequence:
    #
    #   Once you call attnlrp_forward_pass() for the first time, every
    #   later gradient in this process --- including any subsequent
    #   GMAR / GMAR++ / LeGrad / Chefer calls on the SAME Custom_model
    #   instance (or any other instance, since the patch is module-wide)
    #   --- will also be computed via AttnLRP-corrected backward rules,
    #   not vanilla autograd.
    #
    # Recommended usage: run get_score_attnlrp.py as its own separate
    # process/script, not interleaved with your GMAR/LeGrad/Chefer scoring
    # runs in the same Python session.

    def _create_tensor_hook_AttnLRP(self, layer_idx):
        """
        AttnLRP hook: captures relevance flowing into each attention
        tensor for head importance scores.

        After lxt's monkey_patch is applied, the value that arrives here
        via .grad is AttnLRP relevance (Input*Gradient formulation),
        propagated correctly through softmax and the Q/K/V matmuls
        inside attention -- not a plain autograd gradient.
        """
        def hook(grad):
            # grad shape: [Batch, Heads, Tokens, Tokens]
            # Relevance can be signed (positive/negative contribution),
            # so we take abs() before aggregating, consistent with the
            # GMAR hook's convention.
            head_scores = grad.abs().sum(dim=[-2, -1])
            mean_head_scores = head_scores.mean(dim=0)
            self.attention_gradients[layer_idx] = mean_head_scores.detach().cpu().numpy()
            return grad
        return hook

    def attnlrp_forward_pass(self, inputs, target_class=None):
        """
        AttnLRP implementation tracking layer-wise head importance scores.

        Uses lxt's efficient monkey-patch on modeling_vit so that softmax
        and the Q@K^T / Attn@V matmuls inside self-attention propagate
        LRP relevance correctly during the backward pass (this is exactly
        what distinguishes AttnLRP from the Linear-only Gamma-rule LRP in
        lrp_forward_pass above).

        The patch is applied once per process (module-level guard) since
        lxt does not provide a clean way to reverse it -- see the class
        docstring block above this method for the implication that this
        should NOT be interleaved with GMAR/GMAR++/LeGrad/Chefer calls in
        the same process.
        """
        global _ATTNLRP_PATCHED
        if not _ATTNLRP_PATCHED:
            lxt_monkey_patch(modeling_vit, verbose=False)
            _ATTNLRP_PATCHED = True

        self.clear()

        output = self.model(inputs, output_attentions=True)
        logits = output.logits
        native_attentions = output.attentions

        for layer_idx, attn_tensor in enumerate(native_attentions):
            self.attentions.append(attn_tensor.detach().cpu())

            attn_tensor.retain_grad()
            attn_tensor.register_hook(self._create_tensor_hook_AttnLRP(layer_idx))

        if target_class is None:
            target_class = logits.argmax(dim=-1)

        batch_indices = torch.arange(logits.size(0), device=logits.device)
        target_scores = logits[batch_indices, target_class]
        loss_scalar = target_scores.sum()

        self.model.zero_grad()
        loss_scalar.backward()

        return output, self.attentions, self.attention_gradients