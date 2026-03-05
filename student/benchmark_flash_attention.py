import torch
import triton
import itertools
import math

from .a1_basics.model import scaled_dot_product_attention
from .flash_triton import FlashAttention2ForwardTriton

def standard_pytorch_attention(Q, K, V, is_causal=True):
    """标准的 PyTorch 实现，用于作为 Baseline 对比"""

    O = scaled_dot_product_attention(Q, K, V, is_causal=is_causal)
    return O

    # B, N_q, D = Q.shape
    # scale = 1.0 / math.sqrt(D)
    # S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    # if is_causal:
    #     row_idx = torch.arange(N_q, device=Q.device).view(-1, 1)
    #     col_idx = torch.arange(N_q, device=Q.device).view(1, -1)
    #     mask = col_idx > row_idx
    #     S = torch.where(mask, float('-inf'), S)
        
    # P = torch.softmax(S, dim=-1)
    # O = torch.matmul(P, V)
    # return O

def run_benchmark():
    # 参数扫描空间
    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    dims = [16, 32, 64, 128]
    dtypes = [torch.float32, torch.bfloat16]
    batch_size = 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Benchmarking must be run on a CUDA-enabled GPU.")
        
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print("-" * 125)
    print(f"{'Seq_Len':<10} | {'Dim':<5} | {'Dtype':<10} | {'PyTorch Fwd (ms)':<16} | {'Triton Fwd (ms)':<15} | {'PyTorch Bwd (ms)':<16} | {'Triton Bwd (ms)':<15} | {'PyTorch E2E (ms)':<16} | {'Triton E2E (ms)':<15}")
    print("-" * 125)

    for dtype, D, N in itertools.product(dtypes, dims, seq_lens):
        # 初始化张量，设置 requires_grad=True 以进行反向传播
        Q = torch.randn(batch_size, N, D, device=device, dtype=dtype, requires_grad=True)
        K = torch.randn(batch_size, N, D, device=device, dtype=dtype, requires_grad=True)
        V = torch.randn(batch_size, N, D, device=device, dtype=dtype, requires_grad=True)
        dO = torch.randn(batch_size, N, D, device=device, dtype=dtype)
        
        # 结果占位符
        pt_fwd, pt_bwd, pt_e2e = "OOM", "OOM", "OOM"
        tr_fwd, tr_bwd, tr_e2e = "OOM", "OOM", "OOM"

        # ---------------------------------------------------------------------
        # 1. Standard PyTorch Benchmarking
        # ---------------------------------------------------------------------
        try:
            # Forward
            pt_fwd = triton.testing.do_bench(lambda: standard_pytorch_attention(Q, K, V, is_causal=True))
            
            # 预热获取输出用于 Backward
            O_pt = standard_pytorch_attention(Q, K, V, is_causal=True)
            
            # Backward
            pt_bwd = triton.testing.do_bench(lambda: O_pt.backward(dO, retain_graph=True))
            
            # End-to-End
            def pt_e2e_func():
                out = standard_pytorch_attention(Q, K, V, is_causal=True)
                out.backward(dO)
            pt_e2e = triton.testing.do_bench(pt_e2e_func)
            
            pt_fwd = f"{pt_fwd:.3f}"
            pt_bwd = f"{pt_bwd:.3f}"
            pt_e2e = f"{pt_e2e:.3f}"
            
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                pt_fwd, pt_bwd, pt_e2e = "ERR", "ERR", "ERR"
        
        # 清理 PyTorch 缓存，防止影响 Triton 测试
        Q.grad, K.grad, V.grad = None, None, None
        torch.cuda.empty_cache()

        # ---------------------------------------------------------------------
        # 2. Triton FlashAttention Benchmarking
        # ---------------------------------------------------------------------
        try:
            # 假定 FlashAttention2ForwardTriton 是你的实现
            # Forward
            tr_fwd = triton.testing.do_bench(lambda: FlashAttention2ForwardTriton.apply(Q, K, V, True))
            
            O_tr = FlashAttention2ForwardTriton.apply(Q, K, V, True)
            
            # Backward
            tr_bwd = triton.testing.do_bench(lambda: O_tr.backward(dO, retain_graph=True))
            
            # End-to-End
            def tr_e2e_func():
                out = FlashAttention2ForwardTriton.apply(Q, K, V, True)
                out.backward(dO)
            tr_e2e = triton.testing.do_bench(tr_e2e_func)
            
            tr_fwd = f"{tr_fwd:.3f}"
            tr_bwd = f"{tr_bwd:.3f}"
            tr_e2e = f"{tr_e2e:.3f}"
            
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                tr_fwd, tr_bwd, tr_e2e = "ERR", "ERR", "ERR"
                print(f"Triton Error: {e}") # 捕捉 Triton 编译或运行错误

        Q.grad, K.grad, V.grad = None, None, None
        torch.cuda.empty_cache()

        # ---------------------------------------------------------------------
        # 打印输出
        # ---------------------------------------------------------------------
        dtype_str = "fp32" if dtype == torch.float32 else "bf16"
        print(f"{N:<10} | {D:<5} | {dtype_str:<10} | {pt_fwd:<16} | {tr_fwd:<15} | {pt_bwd:<16} | {tr_bwd:<15} | {pt_e2e:<16} | {tr_e2e:<15}")

if __name__ == "__main__":
    run_benchmark()