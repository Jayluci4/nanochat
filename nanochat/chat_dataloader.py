"""
Simple dataloader for chat tasks that handles tokenization.
"""

import torch


def tokenize_conversation(conversation, tokenizer, max_seq_len=2048, pad_id=-100):
    """
    Tokenize a conversation dict into input_ids and targets.

    Args:
        conversation: Dict with 'messages' key containing list of role/content dicts
        tokenizer: The tokenizer instance
        max_seq_len: Maximum sequence length
        pad_id: Padding token ID (use -100 to be ignored by cross_entropy)

    Returns:
        tuple: (input_ids, targets) as torch tensors
    """
    # Format conversation as text
    text_parts = []
    for msg in conversation['messages']:
        role = msg['role']
        content = msg['content']

        if role == 'user':
            text_parts.append(f"<|user|>{content}<|end|>")
        elif role == 'assistant':
            text_parts.append(f"<|assistant|>{content}<|end|>")

    full_text = ''.join(text_parts)

    # Tokenize
    tokens = tokenizer.encode(full_text)

    # Truncate if too long
    if len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]

    # Convert to tensor
    input_ids = torch.tensor(tokens, dtype=torch.long)

    # Targets are same as inputs (next token prediction)
    targets = input_ids.clone()

    return input_ids, targets


def collate_conversations(batch, tokenizer, max_seq_len=2048, device='cuda'):
    """
    Collate a batch of conversations into padded tensors.

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

    # Tokenize all conversations
    tokenized = [tokenize_conversation(conv, tokenizer, max_seq_len) for conv in batch]

    # Always pad to max_seq_len for consistent tensor shapes
    # This is critical for torch.compile and gradient accumulation
    max_len = max_seq_len

    # Padding IDs
    input_pad_id = 0    # Pad input with 0 (or use actual pad token if available)
    target_pad_id = -100  # Pad targets with -100 (ignored by cross_entropy)

    input_ids_list = []
    targets_list = []

    for ids, tgts in tokenized:
        seq_len = len(ids)

        # Pad inputs with 0
        input_padding = torch.full((max_len - seq_len,), input_pad_id, dtype=torch.long)
        padded_ids = torch.cat([ids, input_padding])

        # Pad targets with -100 (ignored by PyTorch's cross_entropy)
        target_padding = torch.full((max_len - seq_len,), target_pad_id, dtype=torch.long)
        padded_tgts = torch.cat([tgts, target_padding])

        input_ids_list.append(padded_ids)
        targets_list.append(padded_tgts)

    # Stack into batch
    input_ids = torch.stack(input_ids_list).to(device)
    targets = torch.stack(targets_list).to(device)

    return input_ids, targets
