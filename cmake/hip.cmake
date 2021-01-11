if(NOT WITH_ROCM)
    return()
endif()

if(NOT DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCm has been installed")
    set(HIP_PATH ${ROCM_PATH}/hip CACHE PATH "Path to which HIP has been installed")
    set(HIP_CLANG_PATH ${ROCM_PATH}/llvm/bin CACHE PATH "Path to which clang has been installed")
else()
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCm has been installed")
    set(HIP_PATH ${ROCM_PATH}/hip CACHE PATH "Path to which HIP has been installed")
    set(HIP_CLANG_PATH ${ROCM_PATH}/llvm/bin CACHE PATH "Path to which clang has been installed")
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

find_package(HIP REQUIRED)
include_directories(${ROCM_PATH}/include)
message(STATUS "HIP version: ${HIP_VERSION}")
message(STATUS "HIP_CLANG_PATH: ${HIP_CLANG_PATH}")

macro(find_package_and_include PACKAGE_NAME)
  find_package("${PACKAGE_NAME}" REQUIRED)
  include_directories("${ROCM_PATH}/${PACKAGE_NAME}/include")
  message(STATUS "${PACKAGE_NAME} version: ${${PACKAGE_NAME}_VERSION}")
endmacro()

find_package_and_include(miopen)
find_package_and_include(rocblas)
find_package_and_include(hiprand)
find_package_and_include(rocrand)
find_package_and_include(rccl)
find_package_and_include(rocthrust)
find_package_and_include(hipcub)
find_package_and_include(rocprim)
find_package_and_include(hipsparse)
find_package_and_include(rocsparse)
find_package_and_include(rocfft)

# add definitions
add_definitions(-DPADDLE_WITH_HIP)
add_definitions(-DEIGEN_USE_GPU)
add_definitions(-DEIGEN_USE_HIP)

# set CXX flags for HIP
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__HIP_PLATFORM_HCC__")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_PLATFORM_HCC__")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP")
set(THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_HIP)

# define HIP_CXX_FLAGS
list(APPEND HIP_CXX_FLAGS -fPIC)
list(APPEND HIP_CXX_FLAGS -D__HIP_PLATFORM_HCC__=1)
# list(APPEND HIP_CXX_FLAGS -D__HIP_NO_HALF_OPERATORS__=1)
list(APPEND HIP_CXX_FLAGS -D__HIP_NO_HALF_CONVERSIONS__=1)
list(APPEND HIP_CXX_FLAGS -Wno-macro-redefined)
list(APPEND HIP_CXX_FLAGS -Wno-inconsistent-missing-override)
list(APPEND HIP_CXX_FLAGS -Wno-exceptions)
list(APPEND HIP_CXX_FLAGS -Wno-shift-count-negative)
list(APPEND HIP_CXX_FLAGS -Wno-shift-count-overflow)
list(APPEND HIP_CXX_FLAGS -Wno-unused-command-line-argument)
list(APPEND HIP_CXX_FLAGS -Wno-duplicate-decl-specifier)
list(APPEND HIP_CXX_FLAGS -Wno-implicit-int-float-conversion)
list(APPEND HIP_CXX_FLAGS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
list(APPEND HIP_CXX_FLAGS -std=c++11)

# set HIP_HIPCC_FLAGS
if(CMAKE_BUILD_TYPE  STREQUAL "Debug")
    list(APPEND HIP_HIPCC_FLAGS  ${CMAKE_CXX_FLAGS_DEBUG})
    list(APPEND HIP_HIPCC_FLAGS -fdebug-info-for-profiling)
elseif(CMAKE_BUILD_TYPE  STREQUAL "RelWithDebInfo")
    list(APPEND HIP_HIPCC_FLAGS  ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
    list(APPEND HIP_HIPCC_FLAGS -fdebug-info-for-profiling)
elseif(CMAKE_BUILD_TYPE  STREQUAL "MinSizeRel")
    list(APPEND HIP_HIPCC_FLAGS  ${CMAKE_CXX_FLAGS_MINSIZEREL})
endif()

# set HIP_CLANG_FLAGS or HIP_HCC_FLAGS
if(HIP_COMPILER STREQUAL clang)
  set(hip_library_name amdhip64)
  set(HIP_CLANG_FLAGS ${HIP_CXX_FLAGS})
  list(APPEND HIP_CLANG_FLAGS -fno-gpu-rdc)
  list(APPEND HIP_CLANG_FLAGS --amdgpu-target=gfx906)
else()
  set(hip_library_name hip_hcc)
  set(HIP_HCC_FLAGS ${HIP_CXX_FLAGS})
  list(APPEND HIP_HCC_FLAGS -fno-gpu-rdc)
  list(APPEND HIP_HCC_FLAGS --amdgpu-target=gfx906)
endif()
message(STATUS "HIP library name: ${hip_library_name}")

# set HIP link libs
find_library(ROCM_HIPRTC_LIB ${hip_library_name} HINTS ${HIP_PATH}/lib)
message(STATUS "ROCM_HIPRTC_LIB: ${ROCM_HIPRTC_LIB}")


# set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -fPIC")
# set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -D__HIP_PLATFORM_HCC__=1")
# set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -D__HIP_NO_HALF_OPERATORS__=1")

# set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} ${HIP_CXX_FLAGS}")
# set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} --amdgpu-target=gfx906")

# include_directories("${ROCM_PATH}/include")
# include_directories("${ROCM_PATH}/hip/include")
# include_directories("${ROCM_PATH}/miopen/include")
# include_directories("${ROCM_PATH}/rocblas/include")
# include_directories("${ROCM_PATH}/hiprand/include")
# include_directories("${ROCM_PATH}/rocrand/include")
# include_directories("${ROCM_PATH}/rccl/include")

# include_directories("${ROCM_PATH}/rocthrust/include/")
# include_directories("${ROCM_PATH}/hipcub/include/")
# include_directories("${ROCM_PATH}/rocprim/include/")
# include_directories("${ROCM_PATH}/hipsparse/include/")
# include_directories("${ROCM_PATH}/rocsparse/include/")
# include_directories("${ROCM_PATH}/rocfft/include/")

# set(HIP_CLANG_PARALLEL_BUILD_COMPILE_OPTIONS "")
# set(HIP_CLANG_PARALLEL_BUILD_LINK_OPTIONS "")
# now default is clang
# set(HIP_COMPILER "clang")

# list(APPEND EXTERNAL_LIBS "-L${ROCM_PATH}/lib/ -lhip_hcc")
# set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -fPIC -DPADDLE_WITH_HIP -DEIGEN_USE_HIP -DEIGEN_USE_GPU -D__HIP_NO_HALF_CONVERSIONS__ -std=c++11 --amdgpu-target=gfx906" )

# if(WITH_RCCL)
#   set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -DPADDLE_WITH_RCCL")
# endif()

# if(NOT WITH_PYTHON)
#   set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -DPADDLE_NO_PYTHON")
# endif(NOT WITH_PYTHON)

# if(WITH_DSO)
#   set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -DPADDLE_USE_DSO")
# endif(WITH_DSO)

# if(WITH_TESTING)
#   set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -DPADDLE_WITH_TESTING")
# endif(WITH_TESTING)

# if(WITH_DISTRIBUTE)
#   set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -DPADDLE_WITH_DISTRIBUTE")
# endif(WITH_DISTRIBUTE)

# if(WITH_GRPC)
#   set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -DPADDLE_WITH_GRPC")
# endif(WITH_GRPC)

# set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -DANY_IMPL_ANY_CAST_MOVEABLE")

# if("${HIP_COMPILER}" STREQUAL "hcc")
#     if("x${HCC_HOME}" STREQUAL "x")
#       set(HCC_HOME "${ROCM_PATH}/hcc")
#     endif()

#     set(CMAKE_HIP_LINK_EXECUTABLE "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HCC_HOME} <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> -ldl --amdgpu-target=gfx906 ")
#     set(CMAKE_HIP_CREATE_SHARED_LIBRARY "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HCC_HOME} <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> -shared --amdgpu-target=gfx906")
#     set(CMAKE_HIP_CREATE_SHARED_MODULE "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HCC_HOME} <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> -shared --amdgpu-target=gfx906")

# elseif("${HIP_COMPILER}" STREQUAL "clang")
    
#     if("x${HIP_CLANG_PATH}" STREQUAL "x")
#         set(HIP_CLANG_PATH  "${ROCM_PATH}/llvm/bin")
#     endif()

#     #Number of parallel jobs by default is 1
#     if(NOT DEFINED HIP_CLANG_NUM_PARALLEL_JOBS)
#       set(HIP_CLANG_NUM_PARALLEL_JOBS 1)
#     endif()
#     #Add support for parallel build and link
#     if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
#       check_cxx_compiler_flag("-parallel-jobs=1" HIP_CLANG_SUPPORTS_PARALLEL_JOBS)
#     endif()
#     if(HIP_CLANG_NUM_PARALLEL_JOBS GREATER 1)
#       if(${HIP_CLANG_SUPPORTS_PARALLEL_JOBS})
#         set(HIP_CLANG_PARALLEL_BUILD_COMPILE_OPTIONS "-parallel-jobs=${HIP_CLANG_NUM_PARALLEL_JOBS} -Wno-format-nonliteral")
#         set(HIP_CLANG_PARALLEL_BUILD_LINK_OPTIONS "-parallel-jobs=${HIP_CLANG_NUM_PARALLEL_JOBS}")
#       else()
#         message("clang compiler doesn't support parallel jobs")
#       endif()
#     endif()


#     # Set the CMake Flags to use the HIP-Clang Compiler.
#     set(CMAKE_HIP_CREATE_SHARED_LIBRARY "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HIP_CLANG_PATH} ${HIP_CLANG_PARALLEL_BUILD_LINK_OPTIONS} <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES> --amdgpu-target=gfx906")
#     set(CMAKE_HIP_CREATE_SHARED_MODULE "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HIP_CLANG_PATH} ${HIP_CLANG_PARALLEL_BUILD_LINK_OPTIONS} <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <LINK_LIBRARIES> -shared --amdgpu-target=gfx906" )
#     set(CMAKE_HIP_LINK_EXECUTABLE "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HIP_CLANG_PATH} ${HIP_CLANG_PARALLEL_BUILD_LINK_OPTIONS} <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> -ldl --amdgpu-target=gfx906")
# endif()
