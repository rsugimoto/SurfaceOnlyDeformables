cmake_minimum_required(VERSION 3.1)
project(main LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -mtune=native")
set(CMAKE_CXX_FLAGS "-Wno-c++11-narrowing -Wno-deprecated")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g2 -ggdb -Wno-c++11-narrowing -Wunknown-pragmas")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
else()
    message(STATUS "CUDA not found. Precomputation will be performed on CPU.")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-xc++>) # force compilation as .cpp files for .cu files
endif()

find_package(OpenMP REQUIRED)

if(OpenMP_FOUND)
    message(STATUS "OpenMP found.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -DVIENNACL_WITH_OPENMP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -DVIENNACL_WITH_OPENMP")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}    ${OpenMP_EXE_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${OpenMP_MODULE_LINKER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")
    set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} ${OpenMP_STATIC_LINKER_FLAGS}")

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp --diag-suppress 20012")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -Xcompiler -Og -g2 -ggdb")
else()
    message(STATUS "OpenMP not found.")
endif()

find_package(Eigen3 REQUIRED)

# Libigl
set(FETCHCONTENT_SOURCE_DIR_LIBIGL ${CMAKE_CURRENT_SOURCE_DIR}/deps/libigl)
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
option(LIBIGL_GLFW "Build target igl::glfw" ON)
option(LIBIGL_PNG "Build target igl::png" ON)
include(libigl)

find_package(MKL)

find_library(OpenCCL NAMES OpenCCL metis HINTS "OpenCCL/Lib")

# Add your project files
file(GLOB SRCFILES *.cpp src/*.cpp src/*.cu)

if(NOT CMAKE_CUDA_COMPILER)
    set_source_files_properties(${SRCFILES} PROPERTIES LANGUAGE CXX)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_executable(${PROJECT_NAME} ${SRCFILES})
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 75-virtual 75-real)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen igl::png igl::glfw)

if(CMAKE_CUDA_COMPILER)
    target_compile_definitions(${PROJECT_NAME} PUBLIC CUDA_AVAILABLE)
endif()

if(OpenMP_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
    target_compile_definitions(${PROJECT_NAME} PUBLIC OMP_AVAILABLE)
endif()

if(MKL_FOUND)
    include_directories(${MKL_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} ${MKL_LIBRARIES})
    target_compile_definitions(${PROJECT_NAME} PUBLIC MKL_AVAILABLE)
endif()

if(OpenCCL)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/OpenCCL/include)
    target_link_directories(${PROJECT_NAME} PUBLIC OpenCCL/Lib)
    target_link_libraries(${PROJECT_NAME} OpenCCL metis)
    target_compile_definitions(${PROJECT_NAME} PUBLIC OPENCCL_AVAILABLE)
    message(STATUS "OpenCCL found.")
endif()
