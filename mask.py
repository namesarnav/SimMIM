
def random_masking(x, mask_ratio):
    """
    Randomly mask patches.

    Args:
        x: (B, N, D)  where N is the number of patches
        mask_ratio: The ratio of patches to mask.

    Returns:
        masked_x: (B, N, D), masked input
        mask: (B, N), mask, 1 is masked, 0 is not masked
    """
    batch_size, seq_len, _ = x.size()
    num_mask = int(mask_ratio * seq_len)
    mask = torch.zeros(batch_size, seq_len, device=x.device)
    for i in range(batch_size):
        mask_indices = torch.randperm(seq_len)[:num_mask]
        mask[i, mask_indices] = 1
    mask = mask.bool()
    masked_x = x.clone()
    masked_x[mask] = 0  # Mask with 0
    return masked_x, mask
