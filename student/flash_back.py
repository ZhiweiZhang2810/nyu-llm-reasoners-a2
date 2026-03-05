
import torch

@staticmethod
def backward(ctx, grad_output):
    # Extract saved tensors from the forward pass
    L, Q, K, V, O = ctx.saved_tensors
    is_causal = getattr(ctx, 'is_causal', False)
        
    # Call the compiled backward function
    dQ, dK, dV = flash_backward_recomputation(
        Q, K, V, O, grad_output, L, is_causal
    )
    
    # Return gradients for Q, K, V, and None for the is_causal flag
    return dQ, dK, dV, None

@torch.compile
def flash_backward_recomputation(Q, K, V, O, dO, L, is_causal=False):
    """
    Computes the backward pass of FlashAttention-2 using recomputation.
    All tensors should have shape (Batch, Seq_Len, Dim), except L which is (Batch, Seq_Len).
    """
    B, N_q, D_dim = Q.shape
    _, N_k, _ = K.shape
    scale = 1.0 / (D_dim ** 0.5)

    # 1. Compute D vector: Row-sum of (dO * O)
    # Shape: (B, N_q, D_dim) -> (B, N_q, 1)
    D_vec = torch.sum(dO * O, dim=-1, keepdim=True)

    # 2. Recompute Scores S
    # Shape: (B, N_q, N_k)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale

    # 3. Apply causal mask to S
    if is_causal:
        row_idx = torch.arange(N_q, device=Q.device).view(-1, 1)
        col_idx = torch.arange(N_k, device=Q.device).view(1, -1)
        mask = col_idx > row_idx
        # Masking with -inf ensures exp(-inf) = 0
        S = torch.where(mask, float('-inf'), S)

    # 4. Recompute P
    # L has shape (B, N_q). Reshape to (B, N_q, 1) for broadcasting over N_k
    L_unsq = L.unsqueeze(-1)
    P = torch.exp(S - L_unsq)

    # 5. Compute dS
    # dP = dO @ V^T -> Shape: (B, N_q, N_k)
    dP = torch.matmul(dO, V.transpose(-2, -1))
    
    # dS = P * (dP - D)
    dS = P * (dP - D_vec)
    
    # Apply causal mask to gradients to prevent backward flow across the diagonal
    if is_causal:
        dS = torch.where(mask, 0.0, dS)
        
    # Apply the attention scale factor to the gradient
    dS = dS * scale

    # 6. Compute dQ, dK, dV
    dQ = torch.matmul(dS, K)
    dK = torch.matmul(dS.transpose(-2, -1), Q)
    dV = torch.matmul(P.transpose(-2, -1), dO)

    return dQ, dK, dV