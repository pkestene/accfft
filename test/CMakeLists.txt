# include dirs
#include_directories(${MPI_INCLUDE_PATH})
include_directories(${FFTW_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# source files list
set(test_dfft_get_local_size_SRCS
  test_dfft_get_local_size.cpp
)

#
# CPU executable
#
add_executable(test_dfft_get_local_size ${test_dfft_get_local_size_SRCS})
target_link_libraries(test_dfft_get_local_size
  accfft
  ${FFTW_LIBRARIES}
)

