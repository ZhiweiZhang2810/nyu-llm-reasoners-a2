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
        L = torch.zeros((B, N_q, 1), device=Q.device, dtype=torch.float32)
        
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
            L[:, i:i+B_q, :] = m_i + torch.log(l_i)
            
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass not required for this step.")