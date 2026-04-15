#pragma once
#include<memory>
#include<string>
#include<sstream>
#include<vector>

template<typename... Args>
inline std::string fmtstr(std::string const& fmt, Args&&... args) {
    int size_s = std::snprintf(nullptr, 0, fmt.c_str(), std::forward<Args>(args)...);
    if (size_s <= 0) {
        throw std::runtime_error("fmtstr: failed to format string");
    }
    auto size = static_cast<size_t>(size_s);
    std::unique_ptr<char[]> buf(new char[size + 1]);
    std::snprintf(buf.get(), size + 1, fmt.c_str(), std::forward<Args>(args)...);
    return std::string(buf.get(), buf.get() + size);
}

inline std::string vec2str(std::vector<std::string> const& vec, std::string const& sep = ", ") {
    std::stringstream ss;
    ss<<"(";
    for (size_t i = 0; i < vec.size() - 1; ++i) {
        ss << vec[i]<<", ";
    }
    ss<< vec.back();
    ss<<")";
    return ss.str();
}

template<typename T>
inline std::string vec2str(std::vector<T> const& vec, std::string const& sep = ", ") {
    std::stringstream ss;
    ss<<"(";
    for (size_t i = 0; i < vec.size() - 1; ++i) {
        ss << vec[i]<<", ";
    }
    ss<< vec.back();
    ss<<")";
    return ss.str();
}

template<typename T>
inline std::string arr2str(T* arr, size_t size, std::string const& sep = ", ") {
    std::stringstream ss;
    ss<<"(";
    for (size_t i = 0; i < size - 1; ++i) {
        ss << arr[i]<<", ";
    }
    ss<< arr[size - 1];
    ss<<")";
    return ss.str();
}