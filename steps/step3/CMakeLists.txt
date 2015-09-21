# optional: use parallel-netcdf to dump data
find_package(PNETCDF)
if (PNETCDF_FOUND)
  add_definitions (-DUSE_PNETCDF)
endif()

# include dirs
include_directories(${FFTW_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# source files list
set (step3_SRCS
  step3.cpp
)

#
# CPU executable
#
add_executable(step3 ${step3_SRCS})
target_link_libraries(step3
  accfft
  accfft_utils
  ${FFTW_LIBRARIES}
)
if (PNETCDF_FOUND)
  target_link_libraries(step3 ${PNETCDF_LIBRARIES})
endif()

#
# GPU executable
#
if(BUILD_GPU)
  set(step3_gpu_SRCS
    step3_gpu.cpp
    )
  
  cuda_add_executable(step3_gpu ${step3_gpu_SRCS})
  set_target_properties(step3_gpu PROPERTIES COMPILE_FLAGS "-DENABLE_GPU")
  target_link_libraries(step3_gpu
    accfft_gpu
    accfft_utils
    ${FFTW_LIBRARIES}
    ${CUDA_LIBRARIES}
  )

  if (PNETCDF_FOUND)
    target_link_libraries(step3_gpu ${PNETCDF_LIBRARIES})
  endif()

  cuda_add_cufft_to_target(step3_gpu)
endif(BUILD_GPU)