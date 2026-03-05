import torch

class FlashAttention2ForwardPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, N_q, D = Q.shape
        _, N_k, _ = K.shape
        
        # Determine tile sizes (at least 16x16)
        B_q, B_k = 32, 32 
        scale = 1.0 / (D ** 0.5)
        
        O = torch.zeros_like(Q)
        L = torch.zeros((B, N_q), device=Q.device, dtype=torch.float32)
        
        # Iterate over query tiles
        for i in range(0, N_q, B_q):
            Q_i = Q[:, i:i+B_q, :] # Shape: (B, B_q, D)
            
            # Initialize running stats for the current query tile
            m_i = torch.full((B, B_q, 1), float('-inf'), device=Q.device)
            l_i = torch.zeros((B, B_q, 1), device=Q.device)
            O_i = torch.zeros_like(Q_i)
            
            # Iterate over key/value tiles
            for j in range(0, N_k, B_k):
                K_j = K[:, j:j+B_k, :]
                V_j = V[:, j:j+B_k, :]
                
                # 1. Compute scores
                S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale
                
                # 2. Causal masking
                if is_causal:
                    row_idx = torch.arange(i, i + B_q, device=Q.device).view(-1, 1)
                    col_idx = torch.arange(j, j + B_k, device=Q.device).view(1, -1)
                    mask = col_idx > row_idx
                    S_ij = torch.where(mask, S_ij - 1e6, S_ij)
                
                # 3. Update running maximum
                m_local = torch.max(S_ij, dim=-1, keepdim=True)[0]
                m_new = torch.maximum(m_i, m_local)
                
                # 4. Compute un-normalized probabilities
                P_ij = torch.exp(S_ij - m_new)
                
                # 5. Rescale and update running sum and output
                alpha = torch.exp(m_i - m_new)
                l_i = l_i * alpha + torch.sum(P_ij, dim=-1, keepdim=True)
                O_i = O_i * alpha + torch.matmul(P_ij, V_j)
                
                m_i = m_new
            
            # 6. Final normalization and LSE calculation
            O[:, i:i+B_q, :] = O_i / l_i
            L[:, i:i+B_q] = (m_i + torch.log(l_i)).squeeze(-1)
            
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

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