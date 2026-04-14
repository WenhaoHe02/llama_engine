#include<algorithm>   // std::fill_n
#include<iostream>    // snprintf
#include<math.h>      // expf, log
#include<stdlib.h>    // rand
#include<string>      // std::string
#include<vector>      // std::vector
#include<random>

#include<cuda.h>
#include<cuda_fp16.h>
#include<cuda_runtime.h>

#include"src/kernels/input_embedding.h"

// there is no embedding cpu kernel implementation now
// `./embedding` to test fp16 GPU kernel
// `./embedding 1` to test fp32 GPU kernel

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

void cpuEmbadding(int* const input_ids, float* output, float* const embadding_table, const int max_context_token_num, const int hidden_size, const int vocab_size) {
    for (int i = 0; i < max_context_token_num; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            output[j + i * hidden_size] = embadding_table[j + input_ids[i] * hidden_size];
        }
    }
}

bool checkResults(float* h_output, float* d_output, const int output_size) {
    float* d_output_cpu = (float*) malloc(output_size * sizeof(float)); // prepare for cpu check
    CHECK(cudaMemcpy(d_output_cpu, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < output_size; ++i) {
        if (fabs(d_output_cpu[i] - h_output[i]) > 1e5) {