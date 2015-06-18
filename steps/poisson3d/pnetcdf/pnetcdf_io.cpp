// for Parallel-netCDF support
#include <pnetcdf.h>
#include <cstdlib>

#define PNETCDF_HANDLE_ERROR {                        \
    if (err != NC_NOERR)                              \
        printf("Error at line %d (%s)\n", __LINE__,   \
               ncmpi_strerror(err));                  \
}

enum ComponentIndex3D {
  IX = 0,
  IY = 1,
  IZ = 2
};


/**
 * Write a parallel-nedcdf file.
 *
 * We assume here that localData is a scalar.
 *
 * Pnetcdf uses row-major format (same as FFTW).
 *
 * \param[in]  filename  : PnetCDF filename
 * \param[in]  starts    : offset to where to start reading data
 * \param[in]  counts    : number of elements read (3D sub-domain inside global)
 * \param[in]  gsizes    : global sizes
 * \param[in]  mpiCoord  : coordinate of current MPI proc cartesian topolgy
 * \param[in]  mpiGrid   : cartesian topo sizes
 * \param[in]  localData : actual data buffer (size : nx*ny*nz*sizeof(double))
 *
 */
void write_pnetcdf(const std::string &filename,
		   MPI_Offset         starts[3],
		   MPI_Offset         counts[3],
		   int                gsizes[3],
		   double            *localData)
{
  int myRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  // netcdf file id
  int ncFileId;
  int err;

  // file creation mode
  int ncCreationMode = NC_CLOBBER;

  // CDF-5 is almost mandatory for very large files (>= 2x10^9 cells)
  // not useful here
  bool useCDF5 = false; 
  if (useCDF5)
    ncCreationMode = NC_CLOBBER|NC_64BIT_DATA;
  else // use CDF-2 file format
    ncCreationMode = NC_CLOBBER|NC_64BIT_OFFSET;

  // verbose log ?
  //bool pnetcdf_verbose = false;
  
  int nbVar=1;
  int dimIds[3], varIds[nbVar];
  //MPI_Offset write_size, sum_write_size;
  MPI_Info mpi_info_used;
  //char str[512];
  
  // time measurement variables
  //double write_timing, max_write_timing, write_bw;

  /* 
   * Create NetCDF file
   */
  err = ncmpi_create(MPI_COMM_WORLD, filename.c_str(), 
		     ncCreationMode,
		     MPI_INFO_NULL, &ncFileId);
  if (err != NC_NOERR) {
    printf("Error: ncmpi_create() file %s (%s)\n",filename.c_str(),ncmpi_strerror(err));
    MPI_Abort(MPI_COMM_WORLD, -1);
    exit(1);
  }

  /*
   * Define global dimensions
   */
  err = ncmpi_def_dim(ncFileId, "x", gsizes[0], &dimIds[0]);
  PNETCDF_HANDLE_ERROR;
  
  err = ncmpi_def_dim(ncFileId, "y", gsizes[1], &dimIds[1]);
  PNETCDF_HANDLE_ERROR;
  
  err = ncmpi_def_dim(ncFileId, "z", gsizes[2], &dimIds[2]);
  PNETCDF_HANDLE_ERROR;

  /* 
   * Define variables to write (give a name)
   */
  nc_type       ncDataType =  NC_DOUBLE;
  MPI_Datatype mpiDataType = MPI_DOUBLE;

  err = ncmpi_def_var(ncFileId, "data", ncDataType, 3, dimIds, &varIds[0]);
  PNETCDF_HANDLE_ERROR;

  /*
   * global attributes
   */
  // did we use CDF-2 or CDF-5
  {
    int useCDF5_int = useCDF5 ? 1 : 0;
    err = ncmpi_put_att_int(ncFileId, NC_GLOBAL, "CDF-5 mode", NC_INT, 1, &useCDF5_int);
    PNETCDF_HANDLE_ERROR;
  }
  
  /* 
   * exit the define mode 
   */
  err = ncmpi_enddef(ncFileId);
  PNETCDF_HANDLE_ERROR;
  
  /* 
   * Get all the MPI_IO hints used
   */
  err = ncmpi_get_file_info(ncFileId, &mpi_info_used);
  PNETCDF_HANDLE_ERROR;
   
  // copy data to write in intermediate buffer
  int nItems = counts[IX]*counts[IY]*counts[IZ];

  {
    
    err = ncmpi_put_vara_all(ncFileId, 
			     varIds[0], 
			     starts, 
			     counts, 
			     localData, 
			     nItems, 
			     mpiDataType);
    PNETCDF_HANDLE_ERROR;
  }


  /* 
   * close the file 
   */
  err = ncmpi_close(ncFileId);
  PNETCDF_HANDLE_ERROR;
  
} // write_pnetcdf
