include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# libcnpy source files list
set(libcnpy_SRCS
  cnpy.cpp
  zlib/crc32.c
  )

# build CPU libaccfft (shared or static ?)
if(BUILD_SHARED_LIBS)
  add_library(cnpy SHARED ${libcnpy_SRCS})
else(BUILD_SHARED_LIBS)
  add_library(cnpy STATIC ${libcnpy_SRCS})
endif(BUILD_SHARED_LIBS)

# build an example
set(cnpy_example_SRCS
  cnpy_example.cpp
  )
add_executable(cnpy_example ${cnpy_example_SRCS})
target_link_libraries(cnpy_example
  cnpy
)
