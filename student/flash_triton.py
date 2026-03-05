import triton
import triton.language as tl
import torch

class FlashAttention2ForwardTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, N_QUERIES, D = Q.shape
        _, N_KEYS, _ = K.shape
        
        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64
        scale = 1.0 / (D ** 0.5)

        O = torch.empty_like(Q)
        L = torch.empty((B, N_QUERIES), device=Q.device, dtype=torch.float32)

        grid = (triton.cdiv(N_QUERIES, Q_TILE_SIZE), B)

        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            scale,
            is_causal=is_causal,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
        )

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_output):
        from .flash_back import backward
        return backward(ctx, grad_output)

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    is_causal: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    # K and V block pointers start at key index 0
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS), # Note: Shape swapped for easy transpose in dot
        strides=(stride_kd, stride_kk), # Strides swapped accordingly
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1),
    )
    
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # Load Q tile
    Q_tile = tl.load(Q_block_ptr)

    # Initialize running statistics in FP32
    m_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    acc = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)

    offs_q = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    # Iterate over key tiles
    for j in range(0, N_KEYS, K_TILE_SIZE):
        K_tile = tl.load(K_block_ptr)
        V_tile = tl.load(V_block_ptr)

        # Q (Q_TILE_SIZE, D) x K (D, K_TILE_SIZE)
        S_ij = tl.dot(Q_tile, K_tile) * scale

        # Apply causal mask
        if is_causal:
            offs_k = j + tl.arange(0, K_TILE_SIZE)
            mask = offs_q[:, None] >= offs_k[None, :]
            S_ij = tl.where(mask, S_ij, S_ij - 1e6)

        # Update running max
        m_local = tl.max(S_ij, axis=1)
        m_new = tl.maximum(m_i, m_local)

        # Rescale factors
        alpha = tl.exp(m_i - m_new)
        P_ij = tl.exp(S_ij - m_new[:, None])

        # Update running sum and accumulators
        l_i = l_i * alpha + tl.sum(P_ij, axis=1)
        
        # Cast P_ij to Value dtype before dot product to match types
        P_ij_casted = P_ij.to(V_tile.dtype)
        acc = acc * alpha[:, None] + tl.dot(P_ij_casted, V_tile)

        # Update global max
        m_i = m_new

        # Advance block pointers along the Key sequence dimension
        K_block_ptr = tl.advance(K_block_ptr, (0, K_TILE_SIZE))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # Final normalization
    acc = acc / l_i[:, None]
    L_out = m_i + tl.log(l_i)

    # Write back to global memory
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    tl.store(O_block_ptr, acc.to(O_ptr.dtype.element_ty))

    # Write Log-Sum-Exp (L)
    L_offs = batch_index * stride_lb + offs_q * stride_lq
    tl.store(L_ptr + L_offs, L_out)