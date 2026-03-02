import contextlib
import timeit
import math
from typing import Callable
import torch
import torch.nn as nn
import a1_basics.model

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def mean(values: list[float]) -> float:
    return sum(values) / len(values)

def stdev(values: list[float]) -> float:
    """Calculate the standard deviation of a list of values."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)

def run_transformer(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    batch_size: int,
    sequence_length: int,
    requires_backward: bool,
    mixed_precision: bool = False,
) -> Callable:
    """Setup and return a function that runs the Transformer forward/(backward) for a single step."""
    model = a1_basics.model.BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    ).to(get_device())

    if mixed_precision and get_device() == "cuda":
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        ctx = contextlib.nullcontext()

    if not requires_backward:
        model.eval()
    else:
        model.train()

    x = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, sequence_length),
        device=get_device(),
        dtype=torch.long
    )

    def run():
        if requires_backward:
            model.zero_grad(set_to_none=True)
            with ctx:
                logits = model(x)
                loss = logits.mean()
            loss.backward()
        else:
            with torch.no_grad(), ctx:
                _ = model(x)
    return run

def benchmark(description: str, run: Callable, num_warmups: int = 5, num_steps: int = 10) -> float:
    """Benchmark `run` by executing it `num_steps` times, return mean time in ms.
    Includes standard deviation calculation.
    """
    is_cuda = torch.cuda.is_available()

    # Warmup
    for _ in range(num_warmups):
        run()
        if is_cuda:
            torch.cuda.synchronize()

    # Time it
    times: list[float] = []
    for step in range(num_steps):
        start_time = timeit.default_timer()
        run()
        if is_cuda:
            torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append((end_time - start_time) * 1000)

    mean_time = mean(times)
    std_time = stdev(times)
    
    print(f"{description}: {mean_time:.2f} ms ± {std_time:.2f} ms (over {num_steps} steps)")
    return mean_time

def print_gpu_specs():
    """Print GPU specifications if available."""
    num_devices = torch.cuda.device_count()
    print(f"\n{num_devices} CUDA device(s) available")
    for i in range(num_devices):
        properties = torch.cuda.get_device_properties(i)
        print(f"  Device {i}: {properties.name}")
        print(f"    Total memory: {properties.total_memory / 1e9:.2f} GB")
        print(f"    Multiprocessors: {properties.multi_processor_count}")

def benchmark_model_sizes():
    """Benchmark Transformer across table sizes and varying context lengths."""
    print("\n" + "=" * 60)
    print("BENCHMARKING TABLE CONFIGURATIONS & CONTEXT LENGTHS")
    print("=" * 60)

    # Dictionary mapped directly from the provided table
    model_configs = {
        "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
        "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
        "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
        "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
        "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    }

    # Parameters to test
    context_lengths = [128, 256, 512, 1024]
    batch_size = 4  # Kept small to mitigate OOM on larger models
    vocab_size = 10000

    for size_name, config in model_configs.items():
        print(f"\n--- Testing Model Size: {size_name.upper()} ---")
        
        for mixed_precision in [True, False]:
            print(f"\n--- Testing Mixed Precision: {mixed_precision} ---")

            for ctx_len in context_lengths:
                desc = f"Size: {size_name:<6} | Ctx/Seq: {ctx_len:<4} | Bsz: {batch_size}"
                
                try:
                    runner = run_transformer(
                        vocab_size=vocab_size,
                        context_length=ctx_len,
                        d_model=config["d_model"],
                        num_layers=config["num_layers"],
                        num_heads=config["num_heads"],
                        d_ff=config["d_ff"],
                        rope_theta=10000.0,
                        batch_size=batch_size,
                        sequence_length=ctx_len, # Match sequence length to context length
                        requires_backward=True,
                        mixed_precision=mixed_precision
                    )
                
                    # Execute benchmark
                    benchmark(desc, runner, num_warmups=3, num_steps=5)
                
                    # Clean up memory explicitly before next iteration to prevent fragmentation
                    del runner
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    # Catch CUDA Out of Memory errors gracefully so the script continues
                    if "out of memory" in str(e).lower():
                        print(f"{desc}: OOM (CUDA Out of Memory)")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        raise e # Re-raise if it's a different runtime error

def main():
    print("Benchmarking BasicsTransformerLM")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("WARNING: Running on CPU. For full experience, use a GPU.")

    print_gpu_specs()

    print("\n--- Sanity check: benchmarking sleep ---")
    import time
    benchmark("sleep(50ms)", lambda: time.sleep(50 / 1000), num_warmups=5, num_steps=10)

    # Run the table configurations
    benchmark_model_sizes()

    print("\n" + "=" * 60)
    print("OBSERVATIONS:")
    print("- Standard Deviation (±): Small values indicate a stable measurement environment.")
    print("- OOM Handling: Larger configurations will skip gracefully if VRAM limits are exceeded.")
    print("=" * 60)

if __name__ == "__main__":
    main()