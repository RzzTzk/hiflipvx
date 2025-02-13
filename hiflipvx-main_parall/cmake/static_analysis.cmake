macro(mn_target_enable_clang_tidy TARGET_NAME VISIBILITY)
    if (NOT CLANG_TIDY_EXE)
        find_program(CLANG_TIDY_EXE NAMES clang-tidy)
        if (CLANG_TIDY_EXE)
            message(STATUS "clang-tidy found at ${CLANG_TIDY_EXE}")
        endif()
    endif()
    if (CLANG_TIDY_EXE)
        set(CMAKE_EXPORT_COMPILE_COMMANDS true)
        if (UNIX)
            execute_process(
                COMMAND bash -c "${CMAKE_CXX_COMPILER} -x c++ -Wp,-v /dev/null 2>&1 > /dev/null | grep '^ /' | grep 'c++'"
                OUTPUT_VARIABLE COMPILER_HEADERS
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            string(REGEX REPLACE "[ \n\t]+" " -I" INCLUDE_COMPILER_HEADERS ${COMPILER_HEADERS})
            separate_arguments(INCLUDE_COMPILER_HEADERS UNIX_COMMAND ${INCLUDE_COMPILER_HEADERS})
            target_compile_options(${TARGET_NAME} ${VISIBILITY} ${INCLUDE_COMPILER_HEADERS})
        endif()

        set(DO_CLANG_TIDY "${CLANG_TIDY_EXE}")
        set_target_properties(${TARGET_NAME} PROPERTIES CXX_CLANG_TIDY "${DO_CLANG_TIDY}" )
        unset(DO_CLANG_TIDY)
    else()
        message(STATUS "clang-tidy not found for ${TARGET_NAME}")
    endif()
endmacro()
