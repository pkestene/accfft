# optional: use parallel-netcdf to dump data
find_package(PNETCDF)
if (PNETCDF_FOUND)
  add_definitions (-DUSE_PNETCDF)
endif()

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
endif()
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

# include dirs
#include_directories(${MPI_INCLUDE_PATH})
include_directories(${FFTW_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# source files list
set(fBm_cpu_SRCS
  fBm_cpu.cpp
)
if (PNETCDF_FOUND)
  set (fBm_cpu_SRCS ${fBm_cpu_SRCS} pnetcdf/pnetcdf_io.cpp)
  include_directories(${PNETCDF_INCLUDES})
endif()


#
# CPU executable
#
add_executable(fBm_cpu ${fBm_cpu_SRCS})
target_link_libraries(fBm_cpu
  accfft
  ${FFTW_LIBRARIES}
)
if (PNETCDF_FOUND)
  target_link_libraries(fBm_cpu ${PNETCDF_LIBRARIES})
endif()

#
# GPU executable
#
if(BUILD_GPU)

  #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-g;-G)

  # source files list
  set(fBm_gpu_SRCS
    fBm_gpu.cpp kernels.cu
    )

  if (PNETCDF_FOUND)
    set (fBm_gpu_SRCS ${fBm_gpu_SRCS} pnetcdf/pnetcdf_io.cpp)
    include_directories(${PNETCDF_INCLUDES})
  endif()

  cuda_add_executable(fBm_gpu ${fBm_gpu_SRCS})
  set_target_properties(fBm_gpu PROPERTIES COMPILE_FLAGS "-DENABLE_GPU")
  target_link_libraries(fBm_gpu
    accfft_gpu
    ${FFTW_LIBRARIES}
    ${CUDA_LIBRARIES}
  )

  if (PNETCDF_FOUND)
    target_link_libraries(fBm_gpu ${PNETCDF_LIBRARIES} ${CUDA_curand_LIBRARY})
  endif()

  cuda_add_cufft_to_target(fBm_gpu)
endif(BUILD_GPU)
