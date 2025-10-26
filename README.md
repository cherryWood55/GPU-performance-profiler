# CUDA Performance Profiler

A lightweight GPU performance profiling tool with real-time metrics collection, bottleneck analysis, and automated reporting.

## Features

- **Real-time GPU Monitoring**: Track GPU utilization, memory bandwidth, and compute usage
- **Kernel Performance Analysis**: Measure execution time, call frequency, and memory consumption
- **Automated Bottleneck Detection**: Identifies performance issues with actionable recommendations
- **Export Capabilities**: Generate CSV reports for visualization and further analysis
- **NVML Integration**: Hardware-level metrics using NVIDIA Management Library

## Requirements

- NVIDIA GPU (Compute Capability 3.0+)
- CUDA Toolkit 11.0+ (includes NVML)
- Linux/Windows with NVIDIA drivers

## Installation

### 1. Clone or download the profiler
```bash
# Save the code as profiler.cu
```

### 2. Compile
```bash
nvcc -O2 profiler.cu -o profiler -lnvidia-ml
```

**Note**: If you get linking errors for `-lnvidia-ml`, try:
```bash
# On Linux
nvcc -O2 profiler.cu -o profiler -L/usr/lib/x86_64-linux-gnu -lnvidia-ml

# Or link statically
nvcc -O2 profiler.cu -o profiler -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lnvidia-ml
```

### 3. Run
```bash
./profiler
```

## Usage

### Basic Profiling

The profiler provides a simple API:

```cpp
GPUProfiler profiler;

// Print GPU hardware info
profiler.printGPUInfo();

// Start profiling a kernel
profiler.startProfile("myKernel");

// Launch and time your kernel
GPUTimer timer;
timer.startTimer();
myKernel<<<blocks, threads>>>(args);
float time_ms = timer.stopTimer();

// Record the results
profiler.recordKernel("myKernel", time_ms);

// Generate reports
profiler.printReport();
profiler.analyzeBottlenecks();
profiler.exportCSV("results.csv");
```

### Example Output

```
=== GPU Information ===
GPU 0: NVIDIA GeForce RTX 3080
  Compute Capability: 8.6
  Total Global Memory: 10240 MB
  Multiprocessors: 68
  Peak Memory Bandwidth: 760.32 GB/s

=== Profiling Report ===
Kernel                     Calls     Total (ms)       Avg (ms)   Memory (MB)      GPU %       Mem %
--------------------------------------------------------------------------------------------------------
vectorAdd                      5          12.450          2.490          381         45.2         32.1
matrixMul                      3        1234.567        411.522         4096         92.3         78.5

=== Bottleneck Analysis ===
Kernel: vectorAdd
  ⚠ Low GPU utilization (45.2%) - Consider increasing parallelism

Kernel: matrixMul
  ⚠ High memory utilization (78.5%) - Memory bandwidth bottleneck
  ⚠ Long execution time (411.522 ms) - Optimize kernel or reduce problem size
```

## Integrating with Your Code

1. **Include the profiler class** in your CUDA project
2. **Wrap your kernel launches** with timing calls
3. **Analyze results** to identify bottlenecks

```cpp
GPUProfiler profiler;
GPUTimer timer;

for (int i = 0; i < iterations; i++) {
    timer.startTimer();
    yourKernel<<<grid, block>>>(args);
    profiler.recordKernel("yourKernel", timer.stopTimer());
}

profiler.printReport();
profiler.analyzeBottlenecks();
```

## Metrics Explained

| Metric | Description | Good Range |
|--------|-------------|------------|
| **Calls** | Number of kernel invocations | N/A |
| **Total (ms)** | Cumulative execution time | N/A |
| **Avg (ms)** | Average time per call | < 10ms for real-time |
| **Memory (MB)** | GPU memory allocated | < 80% of total |
| **GPU %** | Compute utilization | > 70% ideal |
| **Mem %** | Memory bandwidth utilization | 60-80% optimal |

## Bottleneck Indicators

- **Low GPU utilization (< 50%)**: Not enough parallelism, increase block/grid size
- **High memory utilization (> 80%)**: Memory-bound, optimize memory access patterns
- **Long execution time**: Algorithm complexity or inefficient implementation

## Advanced Usage

### Memory Leak Detection

Monitor memory usage across kernel calls:

```cpp
size_t free_mem, total_mem;
profiler.getMemoryUsage(free_mem, total_mem);
std::cout << "Memory used: " << (total_mem - free_mem) / (1024*1024) << " MB\n";
```

### Multi-kernel Comparison

Profile different implementations:

```cpp
profiler.recordKernel("naive_version", time1);
profiler.recordKernel("optimized_version", time2);
profiler.printReport();  // Compare side-by-side
```

## Troubleshooting

### NVML Initialization Failed
```
Error: Failed to initialize NVML
```
**Solution**: Ensure NVIDIA drivers are installed and you have permissions to access GPU metrics.

### Kernel Launch Errors
```
CUDA Error: invalid configuration argument
```
**Solution**: Check your grid/block dimensions don't exceed hardware limits.

### Compilation Errors
```
error: 'threadIdx' was not declared
```
**Solution**: Make sure you're compiling as `.cu` (not `.cpp`) with `nvcc`.

## Limitations

- NVML metrics may require root/admin privileges on some systems
- Profiling overhead: ~0.1-0.5ms per kernel launch
- Does not profile host-to-device transfers (add manually if needed)

## Extending the Profiler

### Add Custom Metrics

```cpp
// In KernelStats struct, add:
float cache_hit_rate;

// In recordKernel(), measure and store
```

### Export to JSON

```cpp
void exportJSON(const std::string& filename) {
    std::ofstream file(filename);
    file << "{\n  \"kernels\": [\n";
    // Add JSON formatting
    file << "  ]\n}\n";
}
```

## Performance Tips

1. **Profile in release mode** (`-O2` or `-O3`)
2. **Run multiple iterations** to get stable averages
3. **Profile representative workloads** not toy examples
4. **Compare against theoretical peak** to gauge efficiency


## Contributing

Suggestions and improvements welcome! Focus areas:
- Additional metric collection
- Better visualization
- Multi-GPU profiling
- Integration with Nsight tools

## References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVML API Documentation](https://docs.nvidia.com/deploy/nvml-api/)
- [GPU Performance Analysis](https://docs.nvidia.com/nsight-compute/)

---
