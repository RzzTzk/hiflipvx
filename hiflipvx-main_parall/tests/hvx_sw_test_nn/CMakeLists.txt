cmake_minimum_required (VERSION 3.20)

project ("hvx_sw_test_nn" 
    VERSION 1.0.0
    DESCRIPTION ""
    LANGUAGES CXX)
add_executable(${PROJECT_NAME} 
    "hvx_sw_test_nn.cpp")
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

#mn_target_enable_clang_tidy(${PROJECT_NAME} PRIVATE)
mn_target_set_default_compile_flags(${PROJECT_NAME})
