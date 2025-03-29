import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

def find_multiple(n: int, k: int) -> int:
    if k == 0 or n % k == 0:
        return n
    return n + k - (n % k)


def pad_weight_(w: nn.Embedding | nn.Linear, multiple: int):
    """Pad the weight of an embedding or linear layer to a multiple of `multiple`."""
    if isinstance(w, nn.Embedding):
        # Pad input dim
        if w.weight.shape[1] % multiple == 0:
            return
        w.weight.data = F.pad(w.weight.data, (0, 0, 0, w.weight.shape[1] % multiple))
        w.num_embeddings, w.embedding_dim = w.weight.shape
    elif isinstance(w, nn.Linear):
        # Pad output dim
        if w.weight.shape[0] % multiple == 0:
            return
        w.weight.data = F.pad(w.weight.data, (0, 0, 0, w.weight.shape[0] % multiple))
        w.out_features, w.in_features = w.weight.shape
    else:
        raise ValueError(f"Unsupported weight type: {type(w)}")


def get_device(prefer: str = "fastest") -> torch.device:
    """
    Select CUDA device based on:
      - Minimum Compute Capability of 7.0 (otherwise fallback to CPU).
      - Preference: 'fastest' or 'memory'.

    Args:
        prefer (str): Device selection preference ('fastest' or 'memory').
                      Default: 'memory'.

    Returns:
        torch.device: Selected device based on preference, or CPU if no
                      suitable GPU is found.
    """

    logger = logging.getLogger("zonos.device")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"CUDA devices available: {device_count}")

        devices_info = []

        for device_idx in range(device_count):
            props = torch.cuda.get_device_properties(device_idx)

            if props.major < 7:
                logger.info(
                    f"Device {device_idx}: {props.name} has Compute Capability "
                    f"{props.major}.{props.minor} < 7.0. Skipping."
                )
                continue

            devices_info.append({
                "idx": device_idx,
                "name": props.name,
                "compute_capability": (props.major, props.minor),
                "multi_processor_count": props.multi_processor_count,
                "total_memory": props.total_memory,
            })

            cc = f"{props.major}.{props.minor}"
            mem_gb = props.total_memory / (1024 ** 3)
            logger.info(
                f"Device {device_idx}: {props.name}, "
                f"Compute Capability: {cc}, "
                f"Multiprocessors: {props.multi_processor_count}, "
                f"Memory: {mem_gb:.2f} GB"
            )

        if not devices_info:
            logger.warning(
                "No devices with Compute Capability >= 7.0 found. "
                "Falling back to CPU."
            )
            return torch.device("cpu")

        if prefer == "memory":
            selected_device = max(devices_info, key=lambda d: d["total_memory"])
        elif prefer == "fastest":
            selected_device = max(
                devices_info,
                key=lambda d: (
                    d["compute_capability"],
                    d["multi_processor_count"],
                    d["total_memory"],
                )
            )
        else:
            raise ValueError(
                f"Unknown preference '{prefer}'. Use 'fastest' or 'memory'."
            )

        logger.info(
            f"Selected device {selected_device['idx']}: {selected_device['name']} "
            f"(CC {selected_device['compute_capability'][0]}."
            f"{selected_device['compute_capability'][1]})"
        )
        return torch.device(f"cuda:{selected_device['idx']}")

    logger.info("No CUDA devices available. Falling back to CPU.")
    return torch.device("cpu")

DEFAULT_DEVICE = get_device("fastest")
