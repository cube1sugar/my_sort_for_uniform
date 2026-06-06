#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <random>
#include <iomanip>
#include <omp.h>
#include <immintrin.h>
#include <execution> // ★追加：並列実行ポリシー用

// ★ Boost
// #include <boost/sort/spreadsort/spreadsort.hpp>

#ifdef _WIN32
#include <malloc.h>
#define ALIGNED_ALLOC(size, align) _aligned_malloc(size, align)
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#define FORCE_INLINE __forceinline
#define RESTRICT __restrict
#else
#include <stdlib.h>
#define ALIGNED_ALLOC(size, align) aligned_alloc(align, size)
#define ALIGNED_FREE(ptr) free(ptr)
#define FORCE_INLINE __attribute__((always_inline)) inline
#define RESTRICT __restrict__
#endif

// --------------------------------------------------------
// DGIS v63 (The Limit Break)
// --------------------------------------------------------
const int BUFFER_SIZE = 8;
#define TICK(x) auto x = std::chrono::high_resolution_clock::now()

// 強制インライン化：関数呼び出しのオーバーヘッドをゼロにする
FORCE_INLINE int get_bucket_index(double val, double min_val, double norm) {
    return (int)((val - min_val) * norm);
}

void dgis_sort_v63_limitbreak(double* RESTRICT data, size_t n, double* RESTRICT output) {
    if (n < 2) { std::memcpy(output, data, n * sizeof(double)); return; }

    const int TARGET_BUCKET_SIZE = 32768;
    int optimal_k = (int)(n / TARGET_BUCKET_SIZE);
    if (optimal_k < 128) optimal_k = 128;
    if (optimal_k > 8192) optimal_k = 8192;
    const int bucket_count = optimal_k;

    // Phase 1: Min/Max (Manual Reduction for MSVC Speed)
    double global_min = 1e300, global_max = -1e300;

#pragma omp parallel
    {
        double local_min = 1e300;
        double local_max = -1e300;
#pragma omp for nowait
        for (long long i = 0; i < (long long)n; i++) {
            double v = data[i];
            if (v < local_min) local_min = v;
            if (v > local_max) local_max = v;
        }
#pragma omp critical
        {
            if (local_min < global_min) global_min = local_min;
            if (local_max > global_max) global_max = local_max;
        }
    }

    if (global_min >= global_max) { std::memcpy(output, data, n * sizeof(double)); return; }

    // 0.999... をかけて、インデックスが bucket_count に到達するのを防ぐ（If文削除のため）
    double norm = (double)(bucket_count * 0.9999999) / (global_max - global_min);
    int num_threads = omp_get_max_threads();

    auto align64 = [](size_t s) { return (s + 63) & ~63; };
    size_t sz_counts = align64(num_threads * bucket_count * sizeof(size_t));
    size_t sz_starts = align64(num_threads * bucket_count * sizeof(size_t));
    size_t sz_cursors = align64(num_threads * bucket_count * sizeof(double*));
    size_t sz_buffers = align64(num_threads * bucket_count * BUFFER_SIZE * sizeof(double));

    // 巨大な単一メモリブロック確保
    unsigned char* arena = (unsigned char*)ALIGNED_ALLOC(sz_counts + sz_starts + sz_cursors + sz_buffers + 4096, 64);
    if (!arena) return;

    // Zero initialization not needed for buffers, only counts
    std::memset(arena, 0, sz_counts);

    size_t* all_local_counts = (size_t*)arena;
    size_t* all_thread_starts = (size_t*)(arena + sz_counts);
    double** all_cursors = (double**)(arena + sz_counts + sz_starts);
    double* all_buffers = (double*)(arena + sz_counts + sz_starts + sz_cursors);

    // Phase 3: Count
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        size_t* my_counts = all_local_counts + tid * bucket_count;

#pragma omp for schedule(static)
        for (long long i = 0; i < (long long)n; i++) {
            // 安全装置（If文）を削除。計算だけでインデックスを決める
            int bucket_idx = (int)((data[i] - global_min) * norm);
            my_counts[bucket_idx]++;
        }
    }

    // Prefix Sum (Serial)
    std::vector<size_t> bucket_starts(bucket_count);
    std::vector<size_t> bucket_sizes(bucket_count);
    {
        size_t current = 0;
        for (int b = 0; b < bucket_count; b++) {
            bucket_starts[b] = current;
            size_t total_count = 0;
            for (int t = 0; t < num_threads; t++) {
                all_thread_starts[t * bucket_count + b] = current;
                size_t c = all_local_counts[t * bucket_count + b];
                current += c;
                total_count += c;
            }
            bucket_sizes[b] = total_count;
        }
    }

    // Phase 4: Scatter (Raw Pointer Speed)
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        size_t* my_starts = all_thread_starts + tid * bucket_count;
        double** my_cursors = all_cursors + tid * bucket_count;
        double* my_buffer_base = all_buffers + tid * (bucket_count * BUFFER_SIZE);

        // カーソル初期化
        for (int b = 0; b < bucket_count; ++b) {
            my_cursors[b] = my_buffer_base + b * BUFFER_SIZE;
        }

#pragma omp for schedule(static)
        for (long long i = 0; i < (long long)n; i++) {
            double val = data[i];
            int bucket_idx = (int)((val - global_min) * norm);

            double* cursor = my_cursors[bucket_idx];
            *cursor = val; // 書き込み

            // ポインタの下位3ビットでバッファ満杯判定 (高速化の肝)
            if (((size_t)(++cursor) & 63) == 0) {
                size_t mem_pos = my_starts[bucket_idx];
                double* src_ptr = cursor - 8;

                // AVX Store
                _mm256_storeu_pd(&output[mem_pos], _mm256_load_pd(src_ptr));
                _mm256_storeu_pd(&output[mem_pos + 4], _mm256_load_pd(src_ptr + 4));

                my_starts[bucket_idx] += 8;
                my_cursors[bucket_idx] = src_ptr; // Reset cursor to start of buffer
            }
            else {
                my_cursors[bucket_idx] = cursor;
            }
        }

        // Flush remaining
        for (int b = 0; b < bucket_count; b++) {
            double* start_ptr = my_buffer_base + b * BUFFER_SIZE;
            double* current_ptr = my_cursors[b];
            size_t count = current_ptr - start_ptr;
            if (count > 0) {
                size_t mem_pos = my_starts[b];
                for (size_t k = 0; k < count; k++) output[mem_pos + k] = start_ptr[k];
            }
        }
    }

    // Phase 5: Local Sort
#pragma omp parallel
    {
        std::vector<uint64_t> radix_buffer;
        radix_buffer.reserve(131072);
        std::vector<size_t> counts(65536);

#pragma omp for schedule(dynamic)
        for (int i = 0; i < bucket_count; i++) {
            size_t size = bucket_sizes[i];
            if (size > 1) {
                double* ptr = output + bucket_starts[i];
                if (size <= 64) {
                    // Small sort
                    for (size_t k = 1; k < size; ++k) {
                        double key = ptr[k];
                        int m = (int)k - 1;
                        while (m >= 0 && ptr[m] > key) {
                            ptr[m + 1] = ptr[m];
                            m--;
                        }
                        ptr[m + 1] = key;
                    }
                    continue;
                }
                // Radix Sort
                if (radix_buffer.size() < size) radix_buffer.resize(size);
                uint64_t* src = (uint64_t*)ptr;
                uint64_t* dst = radix_buffer.data();
                for (int shift = 0; shift < 64; shift += 16) {
                    std::memset(counts.data(), 0, 65536 * sizeof(size_t));
                    for (size_t j = 0; j < size; ++j) {
                        int idx = (src[j] >> shift) & 0xFFFF;
                        counts[idx]++;
                    }
                    size_t accum = 0;
                    for (int j = 0; j < 65536; ++j) {
                        size_t c = counts[j];
                        counts[j] = accum;
                        accum += c;
                    }
                    for (size_t j = 0; j < size; ++j) {
                        int idx = (src[j] >> shift) & 0xFFFF;
                        dst[counts[idx]++] = src[j];
                    }
                    std::swap(src, dst);
                }
                if (src != (uint64_t*)ptr) {
                    std::memcpy(ptr, src, size * sizeof(double));
                }
            }
        }
    }
    ALIGNED_FREE(arena);
}

int main() {
#ifdef _DEBUG
    std::cout << "⚠️ WARNING: DEBUG MODE DETECTED! ⚠️" << std::endl;
    std::cout << "Please use [Ctrl] + [F5] to run in Release mode!" << std::endl;
#endif

    std::vector<size_t> test_cases = {
        300000000, 10000000, 1000000, 200000, 100000
    };

    std::cout << "========================================================================" << std::endl;
    std::cout << "  DGIS vs std::sort (Single Run)" << std::endl;
    std::cout << "========================================================================" << std::endl;
    std::cout << std::left << std::setw(10) << "N"
              << " | " << std::setw(13) << "DGIS Time (s)"
              << " | " << std::setw(7) << "Buckets"
              << " | " << std::setw(13) << "std::sort (s)"
              << " | " << "Speedup" << std::endl;
    std::cout << "----------|---------------|---------|---------------|---------" << std::endl;

    for (size_t N : test_cases) {
        const int TARGET_BUCKET_SIZE = 32768;
        int bucket_count = (int)(N / TARGET_BUCKET_SIZE);
        if (bucket_count < 128) bucket_count = 128;
        if (bucket_count > 8192) bucket_count = 8192;

        double* data_src = (double*)ALIGNED_ALLOC(N * sizeof(double), 64);
        double* data_dgis = (double*)ALIGNED_ALLOC(N * sizeof(double), 64);
        double* data_std = (double*)ALIGNED_ALLOC(N * sizeof(double), 64);

        if (!data_src || !data_dgis || !data_std) {
            std::cout << "\n[Error] Not enough memory!" << std::endl;
            return -1;
        }

        // 乱数生成（ここで全コアがアツアツに起きる）
#pragma omp parallel
        {
            std::mt19937 gen(12345 + omp_get_thread_num());
            std::uniform_real_distribution<> dis(0.0, 100.0);
#pragma omp for
            for (long long i = 0; i < (long long)N; ++i) data_src[i] = dis(gen);
        }

        std::memcpy(data_dgis, data_src, N * sizeof(double));
        std::memcpy(data_std, data_src, N * sizeof(double));

        // DGISの計測（コアが起きている最高な状態でスタート）
        TICK(t1);
        dgis_sort_v63_limitbreak(data_src, N, data_dgis);
        auto t2 = std::chrono::high_resolution_clock::now();
        double time_dgis = std::chrono::duration<double>(t2 - t1).count();

        // std::sortの計測
        TICK(t3);
        std::sort(std::execution::par, data_std, data_std + N); // ★変更：並列化
        auto t4 = std::chrono::high_resolution_clock::now();
        double time_std = std::chrono::duration<double>(t4 - t3).count();

        double speedup = time_std / time_dgis;

        std::cout << std::right << std::setw(9) << N << " | "
                  << std::fixed << std::setprecision(6) << std::setw(13) << time_dgis << " | "
                  << std::right << std::setw(7) << bucket_count << " | "
                  << std::setw(13) << time_std << " | "
                  << std::setprecision(6) << speedup << "x" << std::endl;

        ALIGNED_FREE(data_src);
        ALIGNED_FREE(data_dgis);
        ALIGNED_FREE(data_std);
    }
    return 0;
}
