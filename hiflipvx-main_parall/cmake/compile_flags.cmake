macro(mn_target_set_default_compile_flags TARGET_NAME)
    set(flg_clr_list /W3)
    foreach(flg IN LISTS flg_clr_list)
        string(REPLACE ${flg} "" CMAKE_CXX_FLAGS                   "${CMAKE_CXX_FLAGS}")
        string(REPLACE ${flg} "" CMAKE_CXX_FLAGS_DEBUG             "${CMAKE_CXX_FLAGS_DEBUG}")
        string(REPLACE ${flg} "" CMAKE_CXX_FLAGS_RELEASE           "${CMAKE_CXX_FLAGS_RELEASE}")
    endforeach()
    unset(flg_clr_list)

    target_compile_options(${TARGET_NAME}
        PRIVATE $<$<CONFIG:Debug>:          $<$<CXX_COMPILER_ID:MSVC>: /WX /W4 /TP /MP /JMC /Zc:wchar_t /Zc:__cplusplus>>
        PRIVATE $<$<CONFIG:Release>:        $<$<CXX_COMPILER_ID:MSVC>: /WX /W4 /TP /MP      /Zc:wchar_t /Zc:__cplusplus>>
        PRIVATE $<$<CONFIG:Debug>:          $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>: -Wall -Wextra -Wconversion -pedantic>>
        PRIVATE $<$<CONFIG:Release>:        $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>: -Wall -Wextra -Wconversion -pedantic>>
    )
endmacro()
