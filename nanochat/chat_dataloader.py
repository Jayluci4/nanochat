"""
Simple dataloader for chat tasks that handles tokenization.
Uses tokenizer.render_conversation() to get proper formatting and masking.
"""

import torch


def tokenize_conversation(conversation, tokenizer, max_seq_len=2048):
    """
    Tokenize a conversation using the tokenizer's built-in method.

    Args:
        conversation: Dict with 'messages' key
        tokenizer: The tokenizer instance
        max_seq_len: Maximum sequence length

    Returns:
        tuple: (ids, mask) where mask indicates which tokens to train on
    """
    # Use tokenizer's built-in render_conversation method
    # This handles proper special tokens and masking
    ids, mask = tokenizer.render_conversation(conversation, max_tokens=max_seq_len)

    return ids, mask


def collate_conversations(batch, tokenizer, max_seq_len=2048, device='cuda'):
    """
    Collate a batch of conversations into padded tensors.
    Uses the mask from render_conversation to only train on assistant responses.

    Args:
        batch: List of conversation dicts
        tokenizer: The tokenizer
        max_seq_len: Max sequence length
        device: Device to put tensors on

    Returns:
        tuple: (input_ids, targets) batch tensors
    """
    # Handle single item (not a list)
    if isinstance(batch, dict):
        batch = [batch]

    # Tokenize all conversations (returns ids and mask)
    tokenized = [tokenize_conversation(conv, tokenizer, max_seq_len) for conv in batch]

    # Get pad token from tokenizer
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")

    # Find max length (usually all will be max_seq_len, but handle edge cases)
    max_len = max(len(ids) for ids, mask in tokenized)

    input_ids_list = []
    targets_list = []

    for ids, mask in tokenized:
        n = len(ids)

        # Convert to tensors
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        mask_tensor = torch.tensor(mask, dtype=torch.long)

        # Create inputs and targets (shift by 1 for autoregressive)
        # inputs: [:-1] (all but last)
        # targets: [1:] (all but first), masked where mask==0
        inputs = torch.full((max_len,), pad_token_id, dtype=torch.long)
        targets = torch.full((max_len,), -1, dtype=torch.long)  # -1 = ignore in loss

        # Fill in the actual sequence (shifted)
        inputs[:n-1] = ids_tensor[:-1]

        # Targets are next tokens, but masked by the conversation mask
        target_ids = ids_tensor[1:]
        target_mask = mask_tensor[1:]  # Mask is also shifted
        target_ids[target_mask == 0] = -1  # Ignore where mask is 0
        targets[:n-1] = target_ids

        input_ids_list.append(inputs)
        targets_list.append(targets)

    # Stack into batch
    input_ids = torch.stack(input_ids_list).to(device)
    targets = torch.stack(targets_list).to(device)

    return input_ids, targets
