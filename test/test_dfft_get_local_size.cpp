/**
 *
 * test dfft_get_local_size
 *
 * example:
 * mpirun -np 81 ./test_dfft_get_local_size  --nx 20 --ny 20 --nz 20
 *
 * with the original version (using ceil), multiple process receives isize = 0
 * in at least one dimension.
 */


#include <stdlib.h>
#include <math.h> // for M_PI
#include <mpi.h>

#include "GetPot.h" // for command line arguments

#include <accfft.h>

/******************************************************/
/******************************************************/
/******************************************************/
int main(int argc, char **argv)
{

  int NX,NY,NZ;
  MPI_Init (&argc, &argv);
  int nprocs, procid;
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  /* parse command line arguments */
  GetPot cl(argc, argv);

  NX = cl.follow(128,    "--nx");
  NY = cl.follow(128,    "--ny");
  NZ = cl.follow(128,    "--nz");

  int N[3]={NX,NY,NZ};

  /* Create Cartesian Communicator */
  int c_dims[2];
  MPI_Comm c_comm;
  accfft_create_comm(MPI_COMM_WORLD,c_dims,&c_comm);

  printf("[mpi rank %d] c_dims = %d %d\n", procid, c_dims[0], c_dims[1]);

  int alloc_max=0;
  int isize[3],osize[3],istart[3],ostart[3];
  /* Get the local pencil size and the allocation size */
  alloc_max=accfft_local_size_dft_r2c(N,isize,istart,osize,ostart,c_comm);
  
  {
    bool has_zero_size = (isize[0]*isize[1] == 0);

    /* retrieve MPI Cartesian topology local information */
    int coords[2],np[2],periods[2];
    MPI_Cart_get(c_comm,2,np,periods,coords);
    
    MPI_Barrier(c_comm);
    for(int r=0;r<np[0];r++)
      for(int c=0;c<np[1];c++){
        MPI_Barrier(c_comm);
        if((coords[0]==r) && (coords[1]==c)) {
          if (has_zero_size) {
	    std::cout<<"* ";
	  } else {
	    std::cout<<"  ";
	  }
	  std::cout<<coords[0]<<","<<coords[1]<<" isize[0]= "<<isize[0]<<" isize[1]= "<<isize[1]<<" isize[2]= "<<isize[2]<<" istart[0]= "<<istart[0]<<" istart[1]= "<<istart[1]<<" istart[2]= "<<istart[2]<<std::endl;
	}
        MPI_Barrier(c_comm);
      }
    MPI_Barrier(c_comm);
  }
  
  MPI_Finalize();
  return 0;
} // end main
