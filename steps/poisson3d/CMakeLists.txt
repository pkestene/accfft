# cnpy allow us to dump data in Numpy format:
# easy to read in python, just use numpy.load
add_subdirectory(cnpy)

# optional: use parallel-netcdf to dump data
find_package(PNETCDF)
if (PNETCDF_FOUND)
  add_definitions (-DUSE_PNETCDF)
endif()

# include dirs
#include_directories(${MPI_INCLUDE_PATH})
include_directories(${FFTW_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cnpy)

# source files list
set (poisson3d_cpu_SRCS
  poisson3d_cpu.cpp
)

if (PNETCDF_FOUND)
  set (poisson3d_cpu_SRCS ${poisson3d_cpu_SRCS} pnetcdf/pnetcdf_io.cpp)
  include_directories(${PNETCDF_INCLUDES})
endif()

#
# CPU executable
#
add_executable(poisson3d_cpu ${poisson3d_cpu_SRCS})
target_link_libraries(poisson3d_cpu
  accfft
  ${FFTW_LIBRARIES}
  cnpy
)
if (PNETCDF_FOUND)
  target_link_libraries(poisson3d_cpu ${PNETCDF_LIBRARIES})
endif()

#
# GPU executable
#
if(BUILD_GPU)
  set(poisson3d_gpu_SRCS
    poisson3d_gpu.cpp
    )
  
  if (PNETCDF_FOUND)
    set (poisson3d_gpu_SRCS ${poisson3d_gpu_SRCS} pnetcdf/pnetcdf_io.cpp)
    include_directories(${PNETCDF_INCLUDES})
  endif()

  add_executable(poisson3d_gpu ${poisson3d_gpu_SRCS})
  set_target_properties(poisson3d_gpu PROPERTIES COMPILE_FLAGS "-DENABLE_GPU")
  target_link_libraries(poisson3d_gpu
    accfft_gpu
    ${FFTW_LIBRARIES}
    ${CUDA_LIBRARIES}
    cnpy
  )

  if (PNETCDF_FOUND)
    target_link_libraries(poisson3d_gpu ${PNETCDF_LIBRARIES})
  endif()

  cuda_add_cufft_to_target(poisson3d_gpu)
endif(BUILD_GPU)
