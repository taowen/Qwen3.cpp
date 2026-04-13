cmake_minimum_required(VERSION 3.24)
project(qwen3_cpp_vendor_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(
  EXECUTORCH_PREFIX
  "C:/Apps/qwen3-export/build/executorch-win-vulkan-llm-clangcl/install"
  CACHE PATH "ExecuTorch install prefix"
)

list(APPEND CMAKE_PREFIX_PATH "${EXECUTORCH_PREFIX}")
list(APPEND CMAKE_PREFIX_PATH "${EXECUTORCH_PREFIX}/lib/cmake/ExecuTorch")
list(APPEND CMAKE_PREFIX_PATH "${EXECUTORCH_PREFIX}/lib/cmake/tokenizers")
list(APPEND CMAKE_PREFIX_PATH "${EXECUTORCH_PREFIX}/lib/cmake/re2")
list(APPEND CMAKE_PREFIX_PATH "${EXECUTORCH_PREFIX}/lib/cmake/pcre2")

find_package(executorch CONFIG REQUIRED)

if(NOT TARGET tokenizers::tokenizers)
  message(FATAL_ERROR "tokenizers::tokenizers target not found. Check EXECUTORCH_PREFIX.")
endif()

add_library(vendor_llama_runner STATIC
  src/runner.cpp
  src/llama_tiktoken.cpp
)

target_include_directories(vendor_llama_runner PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(vendor_llama_runner PUBLIC
  executorch
  extension_llm_runner
  tokenizers::tokenizers
)

if(MSVC)
  target_compile_options(vendor_llama_runner PRIVATE /utf-8)
endif()

add_executable(qwen3_chatbot src/chatbot_main.cpp)

target_link_libraries(qwen3_chatbot PRIVATE
  vendor_llama_runner
  ${EXECUTORCH_LIBRARIES}
  tokenizers::tokenizers
)

if(MSVC)
  target_compile_options(qwen3_chatbot PRIVATE /utf-8)
endif()
