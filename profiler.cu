#include <cuda_runtime.h>
#include <nvml.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <map>

class GPUProfiler {
private:
    struct KernelStats {
        std::string name;
        int calls;
        double total_time_ms;
        double avg_time_ms;
        size_t memory_used;
        float compute_util;
        float memory_util;
    };
    
    std::map<std::string, KernelStats> kernel_data;
    nvmlDevice_t device;
    bool nvml_initialized;
    
public:
    GPUProfiler() : nvml_initialized(false) {
        nvmlReturn_t result = nvmlInit();
        if (result == NVML_SUCCESS) {
            nvmlDeviceGetHandleByIndex(0, &device);
            nvml_initialized = true;
        }
    }
    
    ~GPUProfiler() {
        if (nvml_initialized) {
            nvmlShutdown();
        }
    }
    
    void printGPUInfo() {
        int device_count;
        cudaGetDeviceCount(&device_count);
        
        std::cout << "\n=== GPU Information ===\n";
        std::cout << "Number of GPUs: " << device_count << "\n\n";
        
        for (int i = 0; i < device_count; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            
            std::cout << "GPU " << i << ": " << prop.name << "\n";
            std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
            std::cout << "  Total Global Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB\n";
            std::cout << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
            std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
            std::cout << "  Max Threads per MP: " << prop.maxThreadsPerMultiProcessor << "\n";
            std::cout << "  Warp Size: " << prop.warpSize << "\n";
            std::cout << "  Memory Clock Rate: " << (prop.memoryClockRate / 1000) << " MHz\n";
            std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
            std::cout << "  L2 Cache Size: " << (prop.l2CacheSize / 1024) << " KB\n";
            std::cout << "  Peak Memory Bandwidth: " 
                      << (2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6) 
                      << " GB/s\n\n";
        }
    }
    
    void getMemoryUsage(size_t& free_mem, size_t& total_mem) {
        cudaMemGetInfo(&free_mem, &total_mem);
    }
    
    float getGPUUtilization() {
        if (!nvml_initialized) return -1.0f;
        
        nvmlUtilization_t utilization;
        nvmlReturn_t result = nvmlDeviceGetUtilizationRates(device, &utilization);
        
        if (result == NVML_SUCCESS) {
            return utilization.gpu;
        }
        return -1.0f;
    }
    
    float getMemoryUtilization() {
        if (!nvml_initialized) return -1.0f;
        
        nvmlUtilization_t utilization;
        nvmlReturn_t result = nvmlDeviceGetUtilizationRates(device, &utilization);
        
        if (result == NVML_SUCCESS) {
            return utilization.memory;
        }
        return -1.0f;
    }
    
    void startProfile(const std::string& kernel_name) {
        if (kernel_data.find(kernel_name) == kernel_data.end()) {
            kernel_data[kernel_name] = {kernel_name, 0, 0.0, 0.0, 0, 0.0f, 0.0f};
        }
    }
    
    void recordKernel(const std::string& kernel_name, double time_ms) {
        size_t free_mem, total_mem;
        getMemoryUsage(free_mem, total_mem);
        
        auto& stats = kernel_data[kernel_name];
        stats.calls++;
        stats.total_time_ms += time_ms;
        stats.avg_time_ms = stats.total_time_ms / stats.calls;
        stats.memory_used = total_mem - free_mem;
        stats.compute_util = getGPUUtilization();
        stats.memory_util = getMemoryUtilization();
    }
    
    void printReport() {
        std::cout << "\n=== Profiling Report ===\n";
        std::cout << std::left << std::setw(25) << "Kernel" 
                  << std::right << std::setw(10) << "Calls"
                  << std::setw(15) << "Total (ms)"
                  << std::setw(15) << "Avg (ms)"
                  << std::setw(15) << "Memory (MB)"
                  << std::setw(12) << "GPU %"
                  << std::setw(12) << "Mem %\n";
        std::cout << std::string(104, '-') << "\n";
        
        for (const auto& [name, stats] : kernel_data) {
            std::cout << std::left << std::setw(25) << name
                      << std::right << std::setw(10) << stats.calls
                      << std::setw(15) << std::fixed << std::setprecision(3) << stats.total_time_ms
                      << std::setw(15) << stats.avg_time_ms
                      << std::setw(15) << (stats.memory_used / (1024*1024))
                      << std::setw(12) << std::setprecision(1) << stats.compute_util
                      << std::setw(12) << stats.memory_util << "\n";
        }
        std::cout << "\n";
    }
    
    void exportCSV(const std::string& filename) {
        std::ofstream file(filename);
        file << "Kernel,Calls,Total_ms,Avg_ms,Memory_MB,GPU_Util,Mem_Util\n";
        
        for (const auto& [name, stats] : kernel_data) {
            file << name << ","
                 << stats.calls << ","
                 << stats.total_time_ms << ","
                 << stats.avg_time_ms << ","
                 << (stats.memory_used / (1024*1024)) << ","
                 << stats.compute_util << ","
                 << stats.memory_util << "\n";
        }
        
        file.close();
        std::cout << "Report exported to " << filename << "\n";
    }
    
    void analyzeBottlenecks() {
        std::cout << "\n=== Bottleneck Analysis ===\n";
        
        for (const auto& [name, stats] : kernel_data) {
            std::cout << "\nKernel: " << name << "\n";
            
            if (stats.compute_util < 50.0f) {
                std::cout << "  ⚠ Low GPU utilization (" << stats.compute_util 
                          << "%) - Consider increasing parallelism\n";
            }
            
            if (stats.memory_util > 80.0f) {
                std::cout << "  ⚠ High memory utilization (" << stats.memory_util 
                          << "%) - Memory bandwidth bottleneck\n";
            }
            
            if (stats.avg_time_ms > 10.0) {
                std::cout << "  ⚠ Long execution time (" << stats.avg_time_ms 
                          << " ms) - Optimize kernel or reduce problem size\n";
            }
            
            // Calculate GFLOPS estimate (rough)
            double gflops = (stats.memory_used / (1024.0 * 1024.0)) / stats.avg_time_ms;
            if (gflops < 10.0) {
                std::cout << "  ⚠ Low computational throughput\n";
            }
        }
    }
};

// Timer utility
class GPUTimer {
private:
    cudaEvent_t start, stop;
    
public:
    GPUTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    
    ~GPUTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void startTimer() {
        cudaEventRecord(start);
    }
    
    float stopTimer() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// Example kernel for testing
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void matrixMul(float* a, float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main() {
    GPUProfiler profiler;
    profiler.printGPUInfo();
    
    // Test 1: Vector Addition
    int vec_size = 10000000;
    size_t vec_bytes = vec_size * sizeof(float);
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, vec_bytes);
    cudaMalloc(&d_b, vec_bytes);
    cudaMalloc(&d_c, vec_bytes);
    
    GPUTimer timer;
    profiler.startProfile("vectorAdd");
    
    for (int i = 0; i < 5; i++) {
        timer.startTimer();
        vectorAdd<<<(vec_size + 255) / 256, 256>>>(d_a, d_b, d_c, vec_size);
        float time = timer.stopTimer();
        profiler.recordKernel("vectorAdd", time);
    }
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // Test 2: Matrix Multiplication
    int mat_size = 1024;
    size_t mat_bytes = mat_size * mat_size * sizeof(float);
    
    float *d_ma, *d_mb, *d_mc;
    cudaMalloc(&d_ma, mat_bytes);
    cudaMalloc(&d_mb, mat_bytes);
    cudaMalloc(&d_mc, mat_bytes);
    
    profiler.startProfile("matrixMul");
    
    dim3 threads(16, 16);
    dim3 blocks((mat_size + 15) / 16, (mat_size + 15) / 16);
    
    for (int i = 0; i < 3; i++) {
        timer.startTimer();
        matrixMul<<<blocks, threads>>>(d_ma, d_mb, d_mc, mat_size);
        float time = timer.stopTimer();
        profiler.recordKernel("matrixMul", time);
    }
    
    cudaFree(d_ma);
    cudaFree(d_mb);
    cudaFree(d_mc);
    
    // Generate reports
    profiler.printReport();
    profiler.analyzeBottlenecks();
    profiler.exportCSV("profile_report.csv");
    
    return 0;
}