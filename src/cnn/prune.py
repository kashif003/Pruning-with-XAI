import torch
import torch_pruning as tp
from .cnn import Custom_model
from thop import profile, clever_format


def prune_cnn_channels(model, layer_names, channels_to_prune_list, device, img_size=224):
    """
    Structured channel pruning for CNNs (ResNet50/18).

    Unlike ViT head pruning, torch_pruning's DependencyGraph automatically
    traces the coupling between a conv's output channels and everything
    downstream that depends on them -- the next conv's input channels,
    the layer's own BatchNorm, and (importantly for ResNet) the residual
    /skip-connection branch when a channel is shared across the add.
    There's no manual "num_heads"-style bookkeeping to update afterward;
    the module's out_channels shrinks in place and every coupled module
    is updated as part of the same pruning group.

    Args:
        model: the Custom_model wrapper OR a raw torchvision model
        layer_names: list of Conv2d layer names (from named_modules()),
                     e.g. ["layer1.0.conv2", "layer2.1.conv1"]
        channels_to_prune_list: list of lists -- channel indices to
                     prune for each corresponding layer in layer_names
        device: torch device string
        img_size: input resolution (224 for standard ImageNet ResNets)
    """
    model.eval()
    target_model = model.model if hasattr(model, 'model') else model

    for i, layer_name in enumerate(layer_names):
        current_layer_channels = channels_to_prune_list[i]
        target_module = dict(target_model.named_modules())[layer_name]

        # Sort reverse is MANDATORY here -- same reasoning as ViT head
        # pruning: removing a lower index first would shift the indices
        # of channels still queued for removal.
        idxs = sorted(current_layer_channels, reverse=True)

        # Build DG for the CURRENT state of the model (channel counts
        # shift after every group.exec(), so rebuild fresh each time)
        example_inputs = torch.randn(1, 3, img_size, img_size).to(device)
        dg = tp.DependencyGraph().build_dependency(target_model, example_inputs=example_inputs)

        group = dg.get_pruning_group(target_module, tp.prune_conv_out_channels, idxs=idxs)

        if dg.check_pruning_group(group):
            group.exec()
            print(f"Layer {layer_name}: Pruned channels {idxs}. Remaining channels: {target_module.out_channels}")
        else:
            print(f"Layer {layer_name}: Skipped -- pruning group check failed (likely a shared skip-connection dependency).")

    return model


def FLOPS_and_PARAMS(model, input_tensor):
    """Calculates and formats FLOPs and Parameters."""
    # We use verbose=False to keep the console clean
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    return flops, params