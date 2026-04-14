#pragma once
#include<vector>
#include<string>
#include<cstdint>
#include<cuda_fp16.h>
enum class BaseWeightsType{
    FP32_W,
    FP16_W,
    INT8_W,
    UNSUPPORTED_W,
};
template<typename T>
inline BaseWeightsType get_weights_type(){
    if (std::is_same<T, float>::value || std::is_same<T, const float>::value){
        return BaseWeightsType::FP32_W;
    }
    else if (std::is_same<T, half>::value || std::is_same<T, const half>::value){
        return BaseWeightsType::FP16_W;
    }
    else if (std::is_same<T, int8_t>::value || std::is_same<T, const int8_t>::value){
        return BaseWeightsType::INT8_W;
    }
    else{
        return BaseWeightsType::UNSUPPORTED_W;
    }
}
template<typename T>
struct BaseWeights{
    BaseWeightsType type;
    std::vector<int> shape;
    T* data;
    T* bias;
};