cmake_minimum_required (VERSION 3.20)

project ("hvx_hw_test_pool_avg" 
    VERSION 1.0.0
    DESCRIPTION "Manual tests for functions in neural_network/nn_pooling.h"
    LANGUAGES CXX)
add_executable(${PROJECT_NAME} 
    "hvx_hw_test_pool_avg.cpp")
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

#mn_target_enable_clang_tidy(${PROJECT_NAME} PRIVATE)
mn_target_set_default_compile_flags(${PROJECT_NAME})
