cmake_minimum_required(VERSION 3.26.4)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90a)

project(_kernels LANGUAGES CUDA CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/pybind ${CMAKE_BINARY_DIR}/pybind11)

file(GLOB PYTORCH_SOURCES "csrc/*.cu")
pybind11_add_module(_kernels MODULE ${PYTORCH_SOURCES})

target_include_directories(_kernels PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/cutlass/include)
target_include_directories(_kernels PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/pybind/include)
target_include_directories(_kernels PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/flashinfer/include)
target_include_directories(_kernels PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_compile_options(_kernels PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-extended-lambda 
        --expt-relaxed-constexpr
        --use_fast_math
        --disable-warnings
    >
    $<$<COMPILE_LANGUAGE:CXX>:-w>
)
target_link_libraries(_kernels PRIVATE ${TORCH_LIBRARIES} Python::Python pybind11::module ${TORCH_PYTHON_LIBRARY})
