import torch


def subsequent_mask(size: int, device: str = "cpu") -> torch.BoolTensor:
    """Mask out subsequent positions."""
    mask = torch.tril(torch.ones(size, size, device=device, dtype=torch.bool)).unsqueeze(0)
    return mask


def get_forward_attention_mask(mask: torch.BoolTensor) -> torch.Tensor:
    """Get forward attention mask"""
    device = mask.device
    seq_len = mask.size(1)
    subsequent = subsequent_mask(seq_len, device)
    forward_mask = mask.unsqueeze(-1) & subsequent
    return forward_mask


def get_backward_attention_mask(mask: torch.BoolTensor) -> torch.Tensor:
    """Get backward attention mask"""
    device = mask.device
    seq_len = mask.size(1)
    subsequent = subsequent_mask(seq_len, device)
    forward_mask = mask.unsqueeze(-1) & subsequent
    return forward_mask.transpose(1, 2)
