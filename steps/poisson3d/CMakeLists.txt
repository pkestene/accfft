# cnpy allow us to dump data in Numpy format:
# easy to read in python, just use numpy.load
add_subdirectory(cnpy)

# include dirs
#include_directories(${MPI_INCLUDE_PATH})
include_directories(${FFTW_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# source files list
set(poisson3d_cpu_SRCS
  poisson3d_cpu.cpp
)

#
# CPU executable
#
add_executable(poisson3d_cpu ${poisson3d_cpu_SRCS})
target_link_libraries(poisson3d_cpu
  accfft
  ${FFTW_LIBRARIES}
)

#
# GPU executable
#
# if(BUILD_GPU)
#   set(poisson3d_gpu_SRCS
#     poisson3d_gpu.cpp
#   )
#   add_executable(poisson3d_gpu ${poisson3d_gpu_SRCS})
#   set_target_properties(poisson3d_gpu PROPERTIES COMPILE_FLAGS "-DENABLE_GPU")
#   target_link_libraries(poisson3d_gpu
#     accfft_gpu
#     ${FFTW_LIBRARIES}
#     ${CUDA_LIBRARIES}
#   )
#   cuda_add_cufft_to_target(poisson3d_gpu)
# endif(BUILD_GPU)
