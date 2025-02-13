cmake_minimum_required (VERSION 3.20)

project ("dfloat_sqrt" 
    VERSION 1.0.0
    DESCRIPTION "Manual tests for functions in ..."
    LANGUAGES CXX)
add_executable(${PROJECT_NAME} 
    "dfloat_sqrt.cpp")
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

#mn_target_enable_clang_tidy(${PROJECT_NAME} PRIVATE)
mn_target_set_default_compile_flags(${PROJECT_NAME})
