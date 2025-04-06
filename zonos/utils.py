import torch
import torch.nn as nn
import torch.nn.functional as F


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


def get_device(prefer: str = "fastest", debug: bool = True) -> torch.device:
    """
    Select CUDA device based on preference: 'fastest' or 'memory'.

    Args:
        prefer (str): Device selection preference ('fastest' or 'memory'). Default: 'fastest'.
        debug (bool): If True, prints debug information about CUDA devices. Default: False.

    Returns:
        torch.device: Selected device based on preference.
    """
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if debug:
            print(f"CUDA devices available: {device_count}")

        devices_info = []

        for device_idx in range(device_count):
            props = torch.cuda.get_device_properties(device_idx)
            devices_info.append({
                "idx": device_idx,
                "name": props.name,
                "compute_capability": (props.major, props.minor),
                "multi_processor_count": props.multi_processor_count,
                "total_memory": props.total_memory,
            })

            if debug:
                cc = f"{props.major}.{props.minor}"
                mem_gb = props.total_memory / (1024 ** 3)
                print(f"Device {device_idx}: {props.name}, Compute Capability: {cc}, "
                      f"Multiprocessors: {props.multi_processor_count}, "
                      f"Memory: {mem_gb:.2f} GB")

        if prefer == "memory":
            # Select GPU with the most memory
            selected_device = max(devices_info, key=lambda d: d["total_memory"])
        elif prefer == "fastest":
            # Select GPU with highest compute capability, multiprocessor count, then memory
            selected_device = max(
                devices_info,
                key=lambda d: (
                    d["compute_capability"],
                    d["multi_processor_count"],
                    d["total_memory"],
                ),
            )
        else:
            raise ValueError(f"Unknown preference '{prefer}'. Use 'fastest' or 'memory'.")

        if debug:
            print(f"Selected device {selected_device['idx']}: {selected_device['name']}")

        return torch.device(f"cuda:{selected_device['idx']}")

    # MPS breaks for whatever reason. Uncomment when it's working.
    # if torch.mps.is_available():
    #     if debug:
    #         print("Selecting MPS device (Apple Silicon).")
    #     return torch.device("mps")

    if debug:
        print("No CUDA devices available. Falling back to CPU.")

    return torch.device("cpu")


DEFAULT_DEVICE = get_device()
