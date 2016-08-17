/*======================================================================================
We solve the 2-d Laplace problem:    A phi = b

 (A phi)((x,y) = [4 phi(x,y) - phi(x+1,y) - phi(x-1,y) - phi(x,y+1)  -phi(x,y+1)]/a^2 + m^2 phi(x,y) = b(x,y)

 Multiply by scale = 1/(4 + h^2 m^2)

     phi  =  (1 - scale*A) phi +  scale*b = phi + res
          =     scale * (phi(x+1,y) + phi(x-1,y) + phi(x,y+1) + phi(x,y+1))  +  h^2 scale*b(x,y)

At present relex iteration does Gauss Seidel. Can change to Gauss Jacobi or Red Black.

ESW 2016-04-13: Switched to Jacobi.
GEM 2016-08-16: Solve the 2-D Laplace Problem on an Icosahedral

======================================================================================*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#define ICOSAHEDRAL_SIDES 10

typedef struct
{
   int L;    // size of L x L grid
   int niter; // iteration between output
   double m;  // stabilizer term
   double dt; // relaxation parameter
   double scale;
   double c_scale;
   struct dir {
     int ul;   // upper left
     int ur;   // upper right
     int ll;   // lower left
     int lr;   // lower right
   } dir;
   int world_size;
   int rank;
   int y;
} param_t;

void relax(double *phi, double *res, double *tmp,  param_t p, int cycle);
double GetResRoot(double *phi, double *res,  param_t p);
inline int modulo(int a, int b);
int is_valid(int a);

int main(int argc, char** argv)
{
  param_t p;

  //----------- START MPI SETUP ---------------------------------
  //
  // Initialize MPI
  MPI_Init(&argc, &argv);

  // Get the number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &p.world_size);

  // Get the rank
  MPI_Comm_rank(MPI_COMM_WORLD, &p.rank);

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  // Print off a hello world message
  printf("Hello world from processor %s, rank %d out of %d processors\n",
          processor_name, p.rank, p.world_size);
  //
  //----------- END MPI SETUP -----------------------------------

  int i, j;
  int ncycle = 0;
  int dim    = 8;
  if (argc > 1) {
    char *pEnd;
    dim = (int)strtol(argv[1], &pEnd, 10);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (p.rank == 0) printf("\nDimension is %i\n", dim);
  MPI_Barrier(MPI_COMM_WORLD);

  // Setup the rest of our parameters
  p.L       = dim;
  p.m       = 0.01;
  p.scale   = 1/(6.0 +  p.m*p.m);   // scale factor for most of the points
  p.c_scale = 1/(5.0 +  p.m*p.m);   // scale factor for the points on the corners
  p.niter   = 10;
  p.dt      = 0.1;
  p.dir.ur  = modulo(p.rank + 1 + p.rank%2, ICOSAHEDRAL_SIDES);
  p.dir.lr  = modulo(p.rank + 2 - p.rank%2, ICOSAHEDRAL_SIDES);
  p.dir.ul  = modulo(p.rank - 1 - p.rank%2, ICOSAHEDRAL_SIDES);
  p.dir.ll  = modulo(p.rank - 2 + p.rank%2, ICOSAHEDRAL_SIDES);

  // printf("Block %i:\n", p.rank);
  // printf("  Upper Left:  %i\n", p.dir.ul);
  // printf("  Upper Right: %i\n", p.dir.ur);
  // printf("  Lower Left:  %i\n", p.dir.ll);
  // printf("  Lower Right: %i\n", p.dir.lr);

  // SAVING BYTES!
  double *phi;
  double *tmp;
  double *b;
  if (p.rank < 2) {
    phi = (double*)malloc(sizeof(double)*((p.L * p.L)+1));
    tmp = (double*)malloc(sizeof(double)*((p.L * p.L)+1));
    b   = (double*)malloc(sizeof(double)*((p.L * p.L)+1));
  }
  else {
    phi = (double*)malloc(sizeof(double)* p.L * p.L);
    tmp = (double*)malloc(sizeof(double)* p.L * p.L);
    b   = (double*)malloc(sizeof(double)* p.L * p.L);
  }

  // initilize
  for(i = 0; i < p.L; i++) {
    for(j = 0; j < p.L; j++) {
      phi[i*p.L + j] = 0.0;
      tmp[i*p.L + j] = 0.0;
      b  [i*p.L + j] = 0.0;
    }
  }
  if (p.rank < 2) {
    phi[p.L*p.L] = 0.0;
    tmp[p.L*p.L] = 0.0;
    b  [p.L*p.L] = 0.0;
  }

  // setting a point source at (or, close to) the center of each side
  // b[(p.L / 2) * p.L + (p.L / 2)] = 10.0;
  if (p.rank < 2) b[p.L*p.L] = 10.0;

  // iterate to solve_____________________________________
  double resmag = 1.0; // not rescaled.
  resmag = GetResRoot(phi,b,p);
  if (p.rank == 0) {
    printf("At the %d cycle the mag residue is %.8e \n",0,resmag);
  }

  while(resmag > 0.00001) {
    ncycle++;
    relax(phi,b,tmp,p,ncycle - 1);
    resmag = GetResRoot(phi,b,p);
    if (p.rank == 0 && ncycle % 1000 == 0) {
      printf("At the %d cycle the mag residue is %.8e \n",ncycle,resmag);
    }
    // if (ncycle == 100) break;
  }

  int x_offset = (   p.rank    / 2) * p.L;
  int y_offset = ((9-p.rank+1) / 2) * p.L;

  printf("Rank %i: (%i, %i)\n", p.rank, x_offset, y_offset);

  char filename[18];
  snprintf(filename, 18, "./data/data_%i.dat", p.rank);
  FILE *file = fopen(filename, "w+");
  for (i=0; i<p.L; i++) {
    for (j=0; j<p.L; j++) {
      fprintf(file, "%i %i %f\n", i+x_offset, j+y_offset, phi[i*p.L + j]);
    }
  }
  fclose(file);

  free(phi);
  free(tmp);
  free(b);

  MPI_Finalize();
  return 0;
}

void relax(double *phi, double *b, double *tmp, param_t p, int cycle)
{
  int i, j, k;

  // Prepare for async send/recv
  //  |------------|---------|------------|---------|
  //  |    RANK    |  SENDS  |  RECEIVES  |  TOTAL  |
  //  |------------|---------|------------|---------|
  //  |    0, 1    |    8    |      6     |   14    |
  //  |------------|---------|------------|---------|
  //  | 2, 3, 8, 9 |    4    |      5     |    9    |
  //  |------------|---------|------------|---------|
  //  | 4, 5, 6, 7 |    5    |      5     |   10    |
  //  |------------|---------|------------|---------|
  unsigned int num_requests = (p.rank < 2) ? 14 : 0;
  if (num_requests == 0) num_requests = (p.rank > 3 && p.rank < 8) ? 10 : 9;

  int requests = 0;
  MPI_Request *request;
  MPI_Status *status;
  request = (MPI_Request*)malloc(sizeof(MPI_Request)*num_requests);
  status  = (MPI_Status*) malloc(sizeof(MPI_Status) *num_requests);

  double *ul_send = (double*)malloc(sizeof(double)*p.L);
  double *ur_send = (double*)malloc(sizeof(double)*p.L);
  double *ll_send = (double*)malloc(sizeof(double)*p.L);
  double *lr_send = (double*)malloc(sizeof(double)*p.L);

  double *ul_recv = (double*)malloc(sizeof(double)*p.L);
  double *ur_recv = (double*)malloc(sizeof(double)*p.L);
  double *ll_recv = (double*)malloc(sizeof(double)*p.L);
  double *lr_recv = (double*)malloc(sizeof(double)*p.L);

  double *recv_extra_tip_edges;
  unsigned int recv_extra_tip_edges_index = 0;
  if (p.rank < 2) recv_extra_tip_edges = (double*)malloc(sizeof(double)*2);

  for(k = 0; k < p.niter; k ++) {
    requests = 0;
    recv_extra_tip_edges_index = 0;

    for(i = 0; i < p.L; i++) {
      ul_send[i] = phi[i*p.L];
      ur_send[i] = phi[i];
      ll_send[i] = phi[(p.L*(p.L-1))+i];
      lr_send[i] = phi[((i+1)*p.L)-1];
    }

    // Send the edges to their neighbors
    MPI_Isend(ul_send, p.L, MPI_DOUBLE, p.dir.ul, 1, MPI_COMM_WORLD, request + requests++);
    MPI_Isend(ur_send, p.L, MPI_DOUBLE, p.dir.ur, 1, MPI_COMM_WORLD, request + requests++);
    MPI_Isend(ll_send, p.L, MPI_DOUBLE, p.dir.ll, 1, MPI_COMM_WORLD, request + requests++);
    MPI_Isend(lr_send, p.L, MPI_DOUBLE, p.dir.lr, 1, MPI_COMM_WORLD, request + requests++);
    // send the tip (top or bottom) out to all the other blocks
    if (p.rank < 2) {
      for(i = 1; i < 5; i++) {
        MPI_Isend(phi + p.L*p.L, 1, MPI_DOUBLE, (2*i)+p.rank, 1, MPI_COMM_WORLD, request + requests++);
      }
    }
    // send the vals surrounding the tip to block 0 or 1
    //   note that block 0 already has values from blocks 2 and 8,
    //   and that block 1 already has values from blocks 3 and 9
    else if (p.rank > 3 && p.rank < 8) {
      // If it's a top block, we're sending out the first val in phi
      // else we're send out the last val in phi
      double *send_val = (p.rank % 2 == 1) ? phi : phi + p.L*p.L - 1;
      MPI_Isend(send_val, 1, MPI_DOUBLE, p.rank % 2, 1, MPI_COMM_WORLD, request + requests++);
    }

    double *tip = (double*)malloc(sizeof(double));
    // receive the edges from neighbors
    MPI_Irecv(ul_recv, p.L, MPI_DOUBLE, p.dir.ul, 1, MPI_COMM_WORLD, request + requests++);
    MPI_Irecv(ur_recv, p.L, MPI_DOUBLE, p.dir.ur, 1, MPI_COMM_WORLD, request + requests++);
    MPI_Irecv(ll_recv, p.L, MPI_DOUBLE, p.dir.ll, 1, MPI_COMM_WORLD, request + requests++);
    MPI_Irecv(lr_recv, p.L, MPI_DOUBLE, p.dir.lr, 1, MPI_COMM_WORLD, request + requests++);
    // receive the top or bottom from block 1 or 0, respectively
    if (p.rank >= 2) {
      MPI_Irecv(tip, 1, MPI_DOUBLE, p.rank % 2, 1, MPI_COMM_WORLD, request + requests++);
    }
    // receive the neighbors to the top/botton
    else {
      *tip = phi[p.L*p.L];
      for(i = 0; i < 2; i++) {
        // rank 0 receives from 4 and 6
        // rank 1 receives from 5 and 7
        MPI_Irecv(recv_extra_tip_edges + recv_extra_tip_edges_index++, 1, MPI_DOUBLE,
                  i*2 + 4 + p.rank, 1, MPI_COMM_WORLD, request + requests++);
      }
    }

    // Do some other work while we wait!
    // Update everything that doesn't depend on buffers.
    for(i = 1; i < p.L-1; i++) {
      for(j = 1; j < p.L-1; j++) {
        tmp[i*p.L + j] =  (1 - p.dt)*phi[i*p.L + j]                      +
                               p.dt *p.scale*(phi[((i-1)*p.L) +  j   ] +    // previous row, same column
                                              phi[((i-1)*p.L) + (j+1)] +    // previous row, next column
                                              phi[( i   *p.L) + (j-1)] +    // same row, previous column
                                              phi[( i   *p.L) + (j+1)] +    // same row, next column
                                              phi[((i+1)*p.L) + (j-1)] +    // next row, previous column
                                              phi[((i+1)*p.L) +  j   ])  +  // next row, same column
                               p.dt *p.scale*b[i*p.L + j];
      }
    }

    // Wait, if sync hasn't finished.
    MPI_Waitall(requests, request, status);

    // NOW UPDATE ALL OF THE EDGES + TOP + BOTTOM
    double sum;
    for(i = 0; i < p.L; i++) {
      //------------ START UPPER LEFT EDGE -------------------
      //
      if (p.rank % 2 == 1) {
        sum = phi[(i*p.L) + 1] + ul_recv[i];
        if (i == 0) {
          sum += *tip         +
                 ur_recv[0]   +
                 phi    [p.L] +
                 ul_recv[1];
        }
        else if (i == p.L - 1) {
          // I not sure if I'm handling the corners the right way.
          // Currently, if uses just the five neighbor points with the
          // "c_scale" scaling factor.
          sum += phi    [ (p.L-2)*p.L     ] +
                 phi    [((p.L-2)*p.L) + 1]  +
                 ll_recv[0];
        }
        else {
          sum += phi    [ (i-1)*p.L     ] +
                 phi    [((i-1)*p.L) + 1] +
                 phi    [ (i+1)*p.L     ] +
                 ul_recv[i + 1];
        }
      }
      else {
        sum = phi[(i*p.L) + 1] + ul_recv[i];
        if (i == 0) {
          sum += ur_recv[0]   +
                 ur_recv[1]   +
                 phi    [p.L] +
                 ul_recv[1];
        }
        else if (i == p.L - 1) {
          sum += phi[ p.L*(p.L-2)     ] +
                 phi[(p.L*(p.L-2)) + 1] +
                //  phi[(p.L*(p.L-1)) + 1] +   // this is the other corner
                 ll_recv[0];

        }
        else {
          sum += phi[ (i-1)*p.L     ] +
                 phi[((i-1)*p.L) + 1] +
                 phi[ (i+1)*p.L     ] +
                 ul_recv[i + 1];
        }
      }

      // if it's a corner, use the c_scale, which uses a divisor of 5
      double scale = (i == p.L - 1) ? p.c_scale : p.scale;
      tmp[i*p.L] = (1 - p.dt)*phi[i*p.L]  +
                        p.dt *scale*sum +
                        p.dt *scale*b[i*p.L];
      //
      //------------ END UPPER LEFT EDGE ---------------------

      //------------ START LOWER RIGHT EDGE ------------------
      //
      if (p.rank % 2 == 1) {
        sum = phi[((i+1)*p.L)-2] + lr_recv[i];
        if (i == 0) {
          // if (p.rank == 1 && cycle == 0) {
          //   printf("target:    : %f\n", phi[((i+1)*p.L)-1]);
          //   printf("upper right: %f\n", ur_recv[p.L-2]);
          //   printf("right      : %f\n", ur_recv[p.L-1]);
          //   printf("lower right: %f\n", lr_recv[i]);
          //   printf("lower left : %f\n", phi[(p.L*2) - 1]);
          //   printf("left       : %f\n", phi[(p.L*2) - 2]);
          //   printf("upper left : %f\n\n", phi[((i+1)*p.L)-2]);
          // }
          sum += ur_recv[p.L-2] +
                 ur_recv[p.L-1] +
                 phi[(p.L*2) - 1] +
                 phi[(p.L*2) - 2];
        }
        else if (i == p.L - 1) {
          sum += phi[(i*p.L)-1] +
                 lr_recv[p.L-2] +
                 ll_recv[p.L-1] +
                 ll_recv[p.L-2];
        }
        else {
          sum += phi[(i*p.L)-1] +
                 lr_recv[i - 1] +
                 phi[((i+2)*p.L)-1] +
                 phi[((i+2)*p.L)-2];
        }
      }
      else {
        sum = phi[((i+1)*p.L)-2] + lr_recv[i];
        if (i == 0) {
          sum += ur_recv[p.L-1]     +
                 lr_recv[1]         +
                 phi[((i+2)*p.L)-1] +
                 phi[((i+2)*p.L)-2];
        }
        else if (i == p.L - 1) {
          sum += phi[(i*p.L)-1] +         // previous row, same col
                 *tip           +
                 lr_recv[p.L-1] +
                 lr_recv[p.L-2];

        }
        else {
          sum += phi[( i   *p.L)-1] +     // previous row, same col
                 lr_recv[i+1]       +
                 phi[((i+2)*p.L)-1] +     // next row, same col
                 phi[((i+2)*p.L)-2];      // next row, previous col
        }
      }
      tmp[((i+1)*p.L)-1] = (1 - p.dt)*phi[((i+1)*p.L)-1]  +
                                p.dt *p.scale*sum         +
                                p.dt *p.scale*b[((i+1)*p.L)-1];
      //
      //------------ END LOWER RIGHT EDGE --------------------

      // THE CORNERS ONLY NEED TO BE DONE ONCE!
      if (i > 0 && i < p.L-1) {
        //------------ START UPPER RIGHT EDGE ------------------
        //
        sum = ur_recv[i  ]       +    // right
              phi    [i+1]       +    // lower right - same row, next column
              phi    [p.L+i]     +    // lower left  - next row, same column
              phi    [p.L+i - 1] +    // left        - next row, previous column
              phi    [i-1];             // same row, previous column
              if (p.rank % 2 == 1) sum += ur_recv[i-1];
              else sum += ur_recv[i+1];

        tmp[i] = (1 - p.dt)*phi[i]      +
                      p.dt *p.scale*sum +
                      p.dt *p.scale*b[i];
        //
        //------------ END UPPER RIGHT EDGE --------------------

        //------------ START LOWER LEFT EDGE -------------------
        //
        sum = phi[(p.L*(p.L-2))+ i    ] +   // previous row, same column
              phi[(p.L*(p.L-2))+ i + 1] +   // previous row, next column
              phi[(p.L*(p.L-1))+ i + 1] +   // same row, next column
              phi[(p.L*(p.L-1))+ i - 1] +   // same row, previous colum
              ll_recv[i-1] +
              ll_recv[i  ];
        tmp[(p.L*(p.L-1))+i] = (1 - p.dt)*phi[(p.L*(p.L-1))+i]  +
                                    p.dt *p.scale*sum           +
                                    p.dt *p.scale*b[(p.L*(p.L-1))+i];
        //
        //------------ END LOWER LEFT EDGE ---------------------
      }
    }

    //---------- DO THE TOP AND THE BOTTOM CORNERS -------------
    //
    if (p.rank < 2) {
      sum = recv_extra_tip_edges[0] + recv_extra_tip_edges[1];
      if (p.rank == 1) {
        sum += phi[0] +
               ul_recv[0] +
               ur_recv[0];
      }
      else if (p.rank == 0) {
        sum += phi[p.L*p.L - 1] +
               ll_recv[p.L-1] +
               lr_recv[p.L-1];
      }
      tmp[p.L * p.L] = (1 - p.dt)*phi[p.L * p.L]  +
                            p.dt *p.c_scale*sum   +
                            p.dt *p.c_scale*b[p.L * p.L];
    }
    //
    //------------ END THE TOP AND BOTTOM CORNERS --------------

    //------- COPY FROM TMP BACK INTO PHI (FOR JACOBI) ---------
    //
    for(i = 0; i < p.L; i++) {
      for(j = 0; j < p.L; j++) {
        phi[i*p.L + j] = tmp[i*p.L + j];
      }
    }
    if (p.rank < 2) phi[p.L*p.L] = tmp[p.L*p.L];
  }
  // if (p.rank == 0 && cycle == 0) printf("--------------\n\n");

  MPI_Barrier(MPI_COMM_WORLD);
  return;
}


double GetResRoot(double *phi, double *b,  param_t p)
{
  int i, j;

  //true residue
  double residue;
  double ResRoot = 0.0;
  double Bmag = 0.0;

  double ResRoot_global = 0.0;
  double Bmag_global = 0.0;

  // Prepare for async send/recv
  //  |------------|---------|------------|---------|
  //  |    RANK    |  SENDS  |  RECEIVES  |  TOTAL  |
  //  |------------|---------|------------|---------|
  //  |    0, 1    |    8    |      6     |   14    |
  //  |------------|---------|------------|---------|
  //  | 2, 3, 8, 9 |    4    |      5     |    9    |
  //  |------------|---------|------------|---------|
  //  | 4, 5, 6, 7 |    5    |      5     |   10    |
  //  |------------|---------|------------|---------|
  unsigned int num_requests = (p.rank < 2) ? 14 : 0;
  if (num_requests == 0) num_requests = (p.rank > 3 && p.rank < 8) ? 10 : 9;

  int requests = 0;
  MPI_Request *request;
  MPI_Status *status;
  request = (MPI_Request*)malloc(sizeof(MPI_Request)*num_requests);
  status  = (MPI_Status*) malloc(sizeof(MPI_Status) *num_requests);

  double *ul_send = (double*)malloc(sizeof(double)*p.L);
  double *ur_send = (double*)malloc(sizeof(double)*p.L);
  double *ll_send = (double*)malloc(sizeof(double)*p.L);
  double *lr_send = (double*)malloc(sizeof(double)*p.L);

  double *ul_recv = (double*)malloc(sizeof(double)*p.L);
  double *ur_recv = (double*)malloc(sizeof(double)*p.L);
  double *ll_recv = (double*)malloc(sizeof(double)*p.L);
  double *lr_recv = (double*)malloc(sizeof(double)*p.L);

  double *recv_extra_tip_edges;
  unsigned int recv_extra_tip_edges_index = 0;
  if (p.rank < 2) recv_extra_tip_edges = (double*)malloc(sizeof(double)*2);

  for(i = 0; i < p.L; i++) {
    ul_send[i] = phi[i*p.L];
    ur_send[i] = phi[i];
    ll_send[i] = phi[(p.L*(p.L-1))+i];
    lr_send[i] = phi[((i+1)*p.L)-1];
  }

  // Send the edges to their neighbors
  MPI_Isend(ul_send, p.L, MPI_DOUBLE, p.dir.ul, 1, MPI_COMM_WORLD, request + requests++);
  MPI_Isend(ur_send, p.L, MPI_DOUBLE, p.dir.ur, 1, MPI_COMM_WORLD, request + requests++);
  MPI_Isend(ll_send, p.L, MPI_DOUBLE, p.dir.ll, 1, MPI_COMM_WORLD, request + requests++);
  MPI_Isend(lr_send, p.L, MPI_DOUBLE, p.dir.lr, 1, MPI_COMM_WORLD, request + requests++);
  // send the tip (top or bottom) out to all the other blocks
  if (p.rank < 2) {
    for(i = 1; i < 5; i++) {
      MPI_Isend(phi + p.L*p.L, 1, MPI_DOUBLE, (2*i)+p.rank, 1, MPI_COMM_WORLD, request + requests++);
    }
  }
  // send the vals surrounding the tip to block 0 or 1
  //   note that block 0 already has values from blocks 2 and 8,
  //   and that block 1 already has values from blocks 3 and 9
  else if (p.rank > 3 && p.rank < 8) {
    // If it's a top block, we're sending out the first val in phi
    // else we're send out the last val in phi
    double *send_val = (p.rank % 2 == 1) ? phi : phi + p.L*p.L - 1;
    MPI_Isend(send_val, 1, MPI_DOUBLE, p.rank % 2, 1, MPI_COMM_WORLD, request + requests++);
  }

  double *tip = (double*)malloc(sizeof(double));
  // receive the edges from neighbors
  MPI_Irecv(ul_recv, p.L, MPI_DOUBLE, p.dir.ul, 1, MPI_COMM_WORLD, request + requests++);
  MPI_Irecv(ur_recv, p.L, MPI_DOUBLE, p.dir.ur, 1, MPI_COMM_WORLD, request + requests++);
  MPI_Irecv(ll_recv, p.L, MPI_DOUBLE, p.dir.ll, 1, MPI_COMM_WORLD, request + requests++);
  MPI_Irecv(lr_recv, p.L, MPI_DOUBLE, p.dir.lr, 1, MPI_COMM_WORLD, request + requests++);
  // receive the top or bottom from block 1 or 0, respectively
  if (p.rank >= 2) {
    MPI_Irecv(tip, 1, MPI_DOUBLE, p.rank % 2, 1, MPI_COMM_WORLD, request + requests++);
  }
  // receive the neighbors to the top/botton
  else {
    *tip = phi[p.L*p.L];
    for(i = 0; i < 2; i++) {
      // rank 0 receives from 4 and 6
      // rank 1 receives from 5 and 7
      MPI_Irecv(recv_extra_tip_edges + recv_extra_tip_edges_index++, 1, MPI_DOUBLE,
                i*2 + 4 + p.rank, 1, MPI_COMM_WORLD, request + requests++);
    }
  }

  // Do some other work while we wait!
  // Update everything that doesn't depend on buffers.
  for(i = 1; i < p.L-1; i++) {
    for(j = 1; j < p.L-1; j++) {
      residue = p.scale* b[i*p.L + j]
                     - phi[i*p.L + j]
                     + p.scale*(phi[((i-1)*p.L) +  j   ] +
                                phi[((i-1)*p.L) + (j+1)] +
                                phi[( i   *p.L) + (j-1)] +
                                phi[( i   *p.L) + (j+1)] +
                                phi[((i+1)*p.L) + (j-1)] +
                                phi[((i+1)*p.L) +  j   ]);

      ResRoot += residue*residue;
      Bmag += b[i*p.L + j]*b[i*p.L + j];
    }
  }

  // Wait, if sync hasn't finished.
  MPI_Waitall(requests, request, status);

  // NOW UPDATE ALL OF THE EDGES + TOP + BOTTOM
  double sum;
  for(i = 0; i < p.L; i++) {
    //------------ START UPPER LEFT EDGE -------------------
    //
    if (p.rank % 2 == 1) {
      sum = phi[(i*p.L) + 1] + ul_recv[i];
      if (i == 0) {
        sum += *tip         +
               ur_recv[0]   +
               phi    [p.L] +
               ul_recv[1];
      }
      else if (i == p.L - 1) {
        // I not sure if I'm handling the corners the right way.
        // Currently, if uses just the five neighbor points with the
        // "c_scale" scaling factor.
        sum += phi    [ (p.L-2)*p.L     ] +
               phi    [((p.L-2)*p.L) + 1]  +
               ll_recv[0];
      }
      else {
        sum += phi    [ (i-1)*p.L     ] +
               phi    [((i-1)*p.L) + 1] +
               phi    [ (i+1)*p.L     ] +
               ul_recv[i + 1];
      }
    }
    else {
      sum = phi[(i*p.L) + 1] + ul_recv[i];
      if (i == 0) {
        sum += ur_recv[0]   +
               ur_recv[1]   +
               phi    [p.L] +
               ul_recv[1];
      }
      else if (i == p.L - 1) {
        sum += phi[ p.L*(p.L-2)     ] +
               phi[(p.L*(p.L-2)) + 1] +
              //  phi[(p.L*(p.L-1)) + 1] +   // this is the other corner
               ll_recv[0];

      }
      else {
        sum += phi[ (i-1)*p.L     ] +
               phi[((i-1)*p.L) + 1] +
               phi[ (i+1)*p.L     ] +
               ul_recv[i + 1];
      }
    }

    // if it's a corner, use the c_scale, which uses a divisor of 5
    double scale = (i == p.L - 1) ? p.c_scale : p.scale;
    residue = scale*b[i*p.L] - phi[i*p.L] + scale*sum;

    ResRoot += residue*residue;
    Bmag += b[i*p.L]*b[i*p.L];
    //
    //------------ END UPPER LEFT EDGE ---------------------

    //------------ START LOWER RIGHT EDGE ------------------
    //
    if (p.rank % 2 == 1) {
      sum = phi[((i+1)*p.L)-2] + lr_recv[i];
      if (i == 0) {
        sum += ur_recv[p.L-2] +
               ur_recv[p.L-1] +
               phi[(p.L*2) - 1] +
               phi[(p.L*2) - 2];
      }
      else if (i == p.L - 1) {
        sum += phi[(i*p.L)-1] +
               lr_recv[p.L-2] +
               ll_recv[p.L-1] +
               ll_recv[p.L-2];
      }
      else {
        sum += phi[(i*p.L)-1] +
               lr_recv[i - 1] +
               phi[((i+2)*p.L)-1] +
               phi[((i+2)*p.L)-2];
      }
    }
    else {
      sum = phi[((i+1)*p.L)-2] + lr_recv[i];
      if (i == 0) {
        sum += ur_recv[p.L-1]     +
               lr_recv[1]         +
               phi[((i+2)*p.L)-1] +
               phi[((i+2)*p.L)-2];
      }
      else if (i == p.L - 1) {
        sum += phi[(i*p.L)-1] +         // previous row, same col
               *tip           +
               lr_recv[p.L-1] +
               lr_recv[p.L-2];

      }
      else {
        sum += phi[( i   *p.L)-1] +     // previous row, same col
               lr_recv[i+1]       +
               phi[((i+2)*p.L)-1] +     // next row, same col
               phi[((i+2)*p.L)-2];      // next row, previous col
      }
    }
    residue = p.scale* b[((i+1)*p.L)-1] - phi[((i+1)*p.L)-1] + p.scale*sum;

    ResRoot += residue*residue;
    Bmag += b[((i+1)*p.L)-1]*b[((i+1)*p.L)-1];
    //
    //------------ END LOWER RIGHT EDGE --------------------

    // THE CORNERS ONLY NEED TO BE DONE ONCE!
    if (i > 0 && i < p.L-1) {
      //------------ START UPPER RIGHT EDGE ------------------
      //
      sum = ur_recv[i  ]         +
            phi    [i+1]         +    // same row, next column
            phi    [p.L+i]       +    // next row, same column
            phi    [p.L + i - 1] +    // next row, previous column
            phi    [i-1];             // same row, previous column
            if (p.rank % 2 == 1) sum += ur_recv[i-1];
            else sum += ur_recv[i+1];

      residue = p.scale* b[i] - phi[i] + p.scale*sum;

      ResRoot += residue*residue;
      Bmag += b[i]*b[i];
      //
      //------------ END UPPER RIGHT EDGE --------------------

      //------------ START UPPER RIGHT EDGE ------------------
      //
      sum = phi[(p.L*(p.L-2))+ i    ] +   // previous row, same column
            phi[(p.L*(p.L-2))+ i + 1] +   // previous row, next column
            phi[(p.L*(p.L-1))+ i + 1] +   // same row, next column
            phi[(p.L*(p.L-1))+ i - 1] +   // same row, previous colum
            ll_recv[i-1] +
            ll_recv[i  ];
      residue = p.scale* b[(p.L*(p.L-1))+i] - phi[(p.L*(p.L-1))+i] + p.scale*sum;

      ResRoot += residue*residue;
      Bmag += b[(p.L*(p.L-1))+i]*b[(p.L*(p.L-1))+i];
      //
      //------------ END LOWER LEFT EDGE ---------------------
    }
  }

  //---------- DO THE TOP AND THE BOTTOM CORNERS -------------
  //
  if (p.rank < 2) {
    sum = recv_extra_tip_edges[0] + recv_extra_tip_edges[1];
    if (p.rank == 1) {
      sum += phi[0] +
             ul_recv[0] +
             ur_recv[0];
    }
    else if (p.rank == 0) {
      sum += phi[p.L*p.L - 1] +
             ll_recv[p.L-1] +
             lr_recv[p.L-1];
    }
    residue = p.c_scale*b[p.L * p.L] - phi[p.L * p.L] + p.c_scale*sum;

    ResRoot += residue*residue;
    Bmag += b[p.L * p.L]*b[p.L * p.L];
  }
  //
  //------------ END THE TOP AND BOTTOM CORNERS --------------

  MPI_Allreduce(&Bmag, &Bmag_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&ResRoot, &ResRoot_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // Normalized true residue
  return sqrt(ResRoot_global)/sqrt(Bmag_global);
}

// a real module function
int modulo(int a, int b)
{
  const int result = a % b;
  return result >= 0 ? result : result + b;
}
