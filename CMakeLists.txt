cmake_minimum_required(VERSION 3.28)

# The current implementation is compiled only for sm_120a. Set and validate the
# architecture before CUDA compiler detection so CMake never injects a generic
# host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  message(FATAL_ERROR
    "NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX CUDA)

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI, server and perplexity evaluator" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS ninfer_link=1)
  set(CMAKE_JOB_POOL_LINK ninfer_link)
endif()

# Tests also exercise product input, logging and protocol adapters.
set(NINFER_BUILD_PRODUCT_SUPPORT OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_PRODUCT_SUPPORT ON)
endif()

if(MSVC)
  # CUDA 13's CCCL (cuda/std) requires the conforming MSVC preprocessor. The
  # flag is forwarded to the cl host compiler nvcc invokes, including for the
  # host stubs it generates. (The 128-byte-aligned TMA descriptor parameters are
  # passed by device pointer rather than by value precisely so MSVC's stub
  # compiler does not reject them.)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor>)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:preprocessor")
  # The sources are UTF-8 without a BOM, and spdlog's bundled fmt hard-fails the build without
  # /utf-8 (static_assert "Unicode support requires compiling with /utf-8"). cl otherwise decodes
  # sources in the machine's ANSI code page, which silently rewrites every non-ASCII literal.
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/utf-8")
  # windows.h defines min/max as macros which clobber std::min/std::max; NOMINMAX
  # suppresses them. UTF8PROC_STATIC keeps the vendored utf8proc header from
  # marking its symbols __declspec(dllimport) since we compile it statically. Both
  # are scoped to C/CXX (no CUDA file needs them) so the CUDA objects keep their
  # cache. The block precedes the dependency includes so the vendored third-party
  # targets are compiled with the same flags.
  add_compile_definitions(
    $<$<COMPILE_LANGUAGE:C,CXX>:NOMINMAX>
    $<$<COMPILE_LANGUAGE:C,CXX>:UTF8PROC_STATIC>)
endif()

include(cmake/Dependencies.cmake)
include(cmake/NinferTargets.cmake)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
