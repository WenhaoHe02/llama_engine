#include "input_embedding.h"
#include<stdio.h>

template<typename T>
__global__ void input_embedding_kernel(int* const input_ids, T*output, 
    T* const embedding_table, int const max_context_token_num, int const hidden_size){
        size_t index = blockDim.x * blockIdx.x + threadIdx.x;
        while (index < max_context_token_num * hidden_size) {
            int id = input_ids[index / hidden_size];
            output[index] = embedding_table[id * hidden_size + index % hidden_size];
        }
        index += blockDim.x * gridDim.x;
}
template<typename T>
void launchInputEmbedding(TensorWrapper<int>* input_ids,  // INT [token num]
                          TensorWrapper<T>* output,       // FP32 [token num, hidden_size] = [token num, 4096]
                          TensorWrapper<T>* embedding_table) { // FP32 [vocal_size, hidden_size] = [65536, 4096]

                            constexpr int block_size = 256;
                            constexpr int grid_size = 2048;
                            constexpr int max_context_token_num = input_ids->shape[0];
                            constexpr int hidden_size = output->shape[1];
                            LLM_CHECK_WITH_INFO(max_context_token_num == input_ids->shape[0], "input ids 1st shape should equal to 1st shape of output");
                            input_embedding_kernel<T><<<grid_size, block_size>>>(input_ids->data, output->data, embadding_table->data, max_context_token_num, hidden_size);
                            #ifdef PRINT_DATA
                            print_data<<<1, 1>>>(output->data);
                            #else
                            #endif
}   
template void launchInputEmbedding(TensorWrapper<int>* input_ids,    
    TensorWrapper<float>* output,       
    EmbeddingWeight<float>* embed_table);
template void launchInputEmbedding(TensorWrapper<int>* input_ids,    
    TensorWrapper<half>* output,       
    EmbeddingWeight<half>* embed_table);
