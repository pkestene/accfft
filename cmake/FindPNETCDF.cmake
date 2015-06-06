include(FindPackageHandleStandardArgs)
include(CheckFunctionExists)

#If environment variable PNETCDF_ROOT is specified, 
# it has same effect as local PNETCDF_ROOT
if( (NOT PNETCDF_ROOT) AND DEFINED ENV{PNETCDF_ROOT} )
  set( PNETCDF_ROOT "$ENV{PNETCDF_ROOT}" )
  MESSAGE(STATUS "PNETCDF_ROOT is set from environment: ${PNETCDF_ROOT}")
endif()


FIND_PATH(PNETCDF_INCLUDES
          NAMES "pnetcdf.h"
          PATHS ${PNETCDF_ROOT}
	  PATH_SUFFIXES "include"
          NO_DEFAULT_PATH)
MESSAGE(STATUS "PNETCDF_INCLUDES found: ${PNETCDF_INCLUDES}")

IF (${PREFER_SHARED})
  FIND_LIBRARY(PNETCDF_LIB
               NAMES pnetcdf
               PATHS ${PNETCDF_ROOT}
	       PATH_SUFFIXES "lib" "lib64"
               NO_DEFAULT_PATH)
ELSE ()
  FIND_LIBRARY(PNETCDF_LIB 
               NAMES pnetcdf libpnetcdf.a
               PATHS ${PNETCDF_ROOT}
	       PATH_SUFFIXES "lib" "lib64"
               NO_DEFAULT_PATH)
ENDIF ()

if(${PNETCDF_LIB} STREQUAL "PNETCDF_LIB-NOTFOUND")
  MESSAGE(STATUS "PNETCDF library not found")
else()
  MESSAGE(STATUS "PNETCDF library found : $PNETCDF_LIB")
endif()

set(PNETCDF_LIBRARIES ${PNETCDF_LIB})

#SET(CMAKE_REQUIRED_LIBRARIES ${PNETCDF_LIBRARY})
CHECK_FUNCTION_EXISTS(ncmpi_get_varn PNETCDF_VARN)
if(PNETCDF_VARN)
  LIST(APPEND PNETCDF_CPPDEFS -DUSE_PNETCDF_VARN)
  LIST(APPEND PNETCDF_CPPDEFS -DUSE_PNETCDF_VARN_ON_READ)
endif()  

# Handle QUIETLY and REQUIRED.
find_package_handle_standard_args(pnetcdf DEFAULT_MSG
  PNETCDF_LIBRARIES PNETCDF_INCLUDES )

mark_as_advanced(PNETCDF_INCLUDES PNETCDF_LIBRARIES PNETCDF_CPPDEFS)
