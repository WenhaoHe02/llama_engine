#pragma once
#include<string>
#include<fstream>
#include<iostream>
#include<cublas_v2.h>

#define CHECK(call) \
do \
    { \
        cudaError_t err = call; \
        if(err != cudaSuccess){ \
           printf("CUDA Error: \n"); \
           printf("Line: %d\n", __LINE__); \
           printf("Error Code: %d\n", err); \
           printf("Error Text: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } \
    while (0)

[[noreturn]] inline void throw_runtime_error( char* const file,  int const line, std::string const& info = ""){
    throw std::runtime_error(std::string("[oneLLM][ERROR] ") + info + " Assertion fail: " + file + ":"
                             + std::to_string(line) + " \n");
}

inline void llm_assert(bool result, char* const file, int const line, std::string const& info = ""){
    if (!result) {
        throw_runtime_error(file, line, info);
    }
}

#define LLM_CHECK(val) llm_assert(val, __FILE__, __LINE__)
#define LLM_CHECK_WITH_INFO(val, info) \
do { \
    bool is_valid_value = (val); \
    if (!is_valid_value) { \
        llm_assert(false, __FILE__, __LINE__, info); \
    } \
} while (0)