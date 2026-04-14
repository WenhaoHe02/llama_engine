#pragma once
#include <cuda_runtime.h>
#include<cuda.h>
#include<cuda_fp16.h>
#include"src/utils/tensor.h"
#include"src/weights/llama/embedding_weights.h"
template<typename T>
void launch_input_embedding(TenorWrapper<T>* input_ids, TensorWrapper<T>* output, EmbeddingWeights<T>* embedding_table);