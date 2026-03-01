import timeit
import math
from typing import Callable
import torch
import torch.nn as nn
from a1_basics.model import BasicsTransformerLM

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
    requires_backward: bool
) -> Callable:
    """Setup and return a function that runs the Transformer forward/(backward) for a single step."""
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    ).to(get_device())

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
            logits = model(x)
            loss = logits.mean()
            loss.backward()
        else:
            with torch.no_grad():
                _ = model(x)
    return run


def benchmark(description: str, run: Callable, num_warmups: int = 3, num_steps: int = 10) -> float:
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
    
    # 格式化输出：均值 ± 标准差
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


def benchmark_transformer_scaling():
    """Benchmark Transformer with different scaling factors."""
    print("\n" + "=" * 60)
    print("BENCHMARKING TRANSFORMER SCALING")
    print("=" * 60)

    base_config = {
        "vocab_size": 10000,
        "batch_size": 4,
        "context_length": 512,
        "d_model": 256,
        "num_layers": 4,
        "num_heads": 4,
        "d_ff": 1024,
        "rope_theta": 10000.0,
        "sequence_length": 128,
        "requires_backward": True
    }

    print(f"\nBase config: d_model={base_config['d_model']}, layers={base_config['num_layers']}, "
          f"batch={base_config['batch_size']}, seq_len={base_config['sequence_length']}")
    
    base_time = benchmark(
        "run_transformer (base, fwd+bwd)",
        run_transformer(**base_config)
    )

    print("\n--- Forward Only vs Forward+Backward ---")
    fwd_config = base_config.copy()
    fwd_config["requires_backward"] = False
    fwd_time = benchmark(
        "run_transformer (forward only)",
        run_transformer(**fwd_config)
    )
    print(f"  Ratio (Fwd+Bwd / Fwd): ~{base_time/fwd_time:.2f}x")

    print("\n--- Scaling number of layers ---")
    for scale in (2, 3, 4):
        config = base_config.copy()
        config["num_layers"] = scale * base_config["num_layers"]
        result = benchmark(
            f"run_transformer({scale}x num_layers)",
            run_transformer(**config)
        )
        print(f"  Expected ~{scale}x, actual ~{result/base_time:.2f}x")

    print("\n--- Scaling batch size ---")
    for scale in (2, 3, 4):
        config = base_config.copy()
        config["batch_size"] = scale * base_config["batch_size"]
        result = benchmark(
            f"run_transformer({scale}x batch_size)",
            run_transformer(**config)
        )
        print(f"  Expected ~{scale}x, actual ~{result/base_time:.2f}x")


def main():
    print("Benchmarking BasicsTransformerLM")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("WARNING: Running on CPU. For full experience, use a GPU.")

    print_gpu_specs()

    print("\n--- Sanity check: benchmarking sleep ---")
    import time
    benchmark("sleep(50ms)", lambda: time.sleep(50 / 1000), num_warmups=2, num_steps=3)

    benchmark_transformer_scaling()

    print("\n" + "=" * 60)
    print("OBSERVATIONS:")
    print("- Standard Deviation (±): Small values indicate a stable measurement environment.")
    print("- Scaling layers: should scale ~linearly.")
    print("- Forward vs Backward: backward pass adds significant overhead.")
    print("=" * 60)


if __name__ == "__main__":
    main()