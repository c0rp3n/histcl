cmake_minimum_required(VERSION 3.10)

# Project name
project(HistCL
    VERSION 0.1.0
    DESCRIPTION ""
    LANGUAGES C CXX
)

find_package(OpenCL REQUIRED)

# Do stuff depending on the compiler
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "-W -Wall -Wextra -Wpedantic -Wunused-value -Wold-style-cast")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "/W4")
    set(CMAKE_CXX_FLAGS_DEBUG "/O0 /ZI")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2")
endif()

add_executable(histcl
    src/compute/engine_cl.cpp
    src/main.cpp
)

set_target_properties(histcl
    PROPERTIES 
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
)

target_include_directories(histcl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(histcl PUBLIC OpenCL)
