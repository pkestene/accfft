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
set (powerSpectrum_cpu_SRCS
  powerSpectrum_cpu.cpp
)

if (PNETCDF_FOUND)
  set (powerSpectrum_cpu_SRCS ${powerSpectrum_cpu_SRCS} pnetcdf/pnetcdf_io.cpp)
  include_directories(${PNETCDF_INCLUDES})
endif()

#
# CPU executable
#
add_executable(powerSpectrum_cpu ${powerSpectrum_cpu_SRCS})
target_link_libraries(powerSpectrum_cpu
  accfft
  ${FFTW_LIBRARIES}
  cnpy
)
if (PNETCDF_FOUND)
  target_link_libraries(powerSpectrum_cpu ${PNETCDF_LIBRARIES})
endif()

#
# GPU executable
#
# if(BUILD_GPU)
#   set(powerSpectrum_gpu_SRCS
#     powerSpectrum_gpu.cpp kernels.cu
#     )
  
#   if (PNETCDF_FOUND)
#     set (powerSpectrum_gpu_SRCS ${powerSpectrum_gpu_SRCS} pnetcdf/pnetcdf_io.cpp)
#     include_directories(${PNETCDF_INCLUDES})
#   endif()

#   cuda_add_executable(powerSpectrum_gpu ${powerSpectrum_gpu_SRCS})
#   set_target_properties(powerSpectrum_gpu PROPERTIES COMPILE_FLAGS "-DENABLE_GPU")
#   target_link_libraries(powerSpectrum_gpu
#     accfft_gpu
#     ${FFTW_LIBRARIES}
#     ${CUDA_LIBRARIES}
#     cnpy
#   )

#   if (PNETCDF_FOUND)
#     target_link_libraries(powerSpectrum_gpu ${PNETCDF_LIBRARIES})
#   endif()

#   cuda_add_cufft_to_target(powerSpectrum_gpu)
# endif(BUILD_GPU)
