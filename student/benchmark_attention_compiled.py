import timeit
import math
import torch
import a1_basics.model

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def mean(values: list[float]) -> float:
    return sum(values) / len(values)

def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)

def run_attention_benchmark(
    batch_size: int,
    seq_len: int,
    d_model: int,
    use_compile: bool = False,
    num_warmups: int = 10,
    num_steps: int = 100
):
    device = get_device()
    is_cuda = device == "cuda"
    
    # Create inputs (No multi-head dimension as requested: [batch, seq, d_model])
    q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    k = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    v = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    
    # Setup compiled or eager function
    attn_fn = a1_basics.model.scaled_dot_product_attention
    if use_compile:
        attn_fn = torch.compile(attn_fn)
    
    # 1. Warmup Forward
    for _ in range(num_warmups):
        out = attn_fn(q, k, v)
        if is_cuda: torch.cuda.synchronize()

    # 2. Time Forward
    fwd_times = []
    for _ in range(num_steps):
        start = timeit.default_timer()
        out = attn_fn(q, k, v)
        if is_cuda: torch.cuda.synchronize()
        fwd_times.append((timeit.default_timer() - start) * 1000)

    # Calculate Forward metrics
    fwd_mean = mean(fwd_times)
    fwd_std = stdev(fwd_times)

    # 3. Measure Memory before Backward (Peak memory of forward activations)
    mem_used_mb = 0.0
    if is_cuda:
        mem_used_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        # We also need a fake loss to trigger backward
        loss = out.mean()

    # 4. Warmup Backward
    # We must retain graph or recompute forward for multiple backward steps
    for _ in range(num_warmups):
        # Recompute forward to create a fresh computational graph for each backward
        out = attn_fn(q, k, v)
        loss = out.mean()
        loss.backward()
        # clear gradients
        q.grad, k.grad, v.grad = None, None, None
        if is_cuda: torch.cuda.synchronize()

    # 5. Time Backward
    bwd_times = []
    for _ in range(num_steps):
        out = attn_fn(q, k, v)
        loss = out.mean()
        
        start = timeit.default_timer()
        loss.backward()
        if is_cuda: torch.cuda.synchronize()
        bwd_times.append((timeit.default_timer() - start) * 1000)
        
        q.grad, k.grad, v.grad = None, None, None

    # Calculate Backward metrics
    bwd_mean = mean(bwd_times)
    bwd_std = stdev(bwd_times)

    return fwd_mean, fwd_std, bwd_mean, bwd_std, mem_used_mb

def main():
    print("Benchmarking Scaled Dot-Product Attention (Eager vs Compiled)")
    print("=" * 85)
    
    batch_size = 8
    d_models = [16, 32, 64, 128]
    seq_lengths = [256, 1024, 4096, 8192, 16384]
    
    # Table header
    print(f"{'d_model':<8} | {'Seq Len':<8} | {'Eager Fwd':<10} | {'Comp Fwd':<10} | {'Eager Bwd':<10} | {'Comp Bwd':<10}")
    print("-" * 85)

    for d_model in d_models:
        for seq_len in seq_lengths:
            try:
                # Eager pass
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
                ef_m, ef_s, eb_m, eb_s, _ = run_attention_benchmark(
                    batch_size=batch_size, seq_len=seq_len, d_model=d_model, use_compile=False
                )
                
                # Compiled pass
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
                cf_m, cf_s, cb_m, cb_s, _ = run_attention_benchmark(
                    batch_size=batch_size, seq_len=seq_len, d_model=d_model, use_compile=True
                )
                
                print(f"{d_model:<8} | {seq_len:<8} | {ef_m:>4.2f}±{ef_s:<4.2f} | {cf_m:>4.2f}±{cf_s:<4.2f} | {eb_m:>4.2f}±{eb_s:<4.2f} | {cb_m:>4.2f}±{cb_s:<4.2f}")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"{d_model:<8} | {seq_len:<8} | {'OOM':<10} | {'OOM':<10} | {'OOM':<10} | {'OOM':<10}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise e

if __name__ == "__main__":
    main()