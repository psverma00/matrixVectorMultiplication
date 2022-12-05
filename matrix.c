#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include "cuda_utils.h"
#include "mkl.h"

void CreatMatVec(double *A, double *C, double *D, long int ROW, long int COLUMN)
{
  for(int i=0; i<ROW; i++)
  {
    C[i] = 1;
    //C[i] = drand48();
    D[i] = 0;    
    for(int j=0; j<COLUMN; j++)
    {	  
      A[i*COLUMN+j] = i*COLUMN+j;
      //A[i*COLUMN+j] = drand48();
    }
  }

}	

void matVecMulti(double *A, double *C, double *D, long int ROW, long int COLUMN)
{
  for(int i=0; i<ROW; i++)
  {
    for(int j=0; j<COLUMN; j++)
    {	  
      D[i] += A[i*COLUMN+j] * C[j];
    }
    //printf(" D[%d] : %lf \n", i, D[i]); 
  }

}	



int main()
{
  srand48(time(NULL));     	
  long int memMat, memVec;
  long int row, column;
  int diff; 
 
  row = 100000;  
  column = 100000;

  memMat = row*column*sizeof(double);
  memVec = row*sizeof(double);

  double *a = (double*)malloc(memMat);
  double *c = (double*)malloc(memVec);
  double *d = (double*)malloc(memVec);
  double *dBlas = (double*)malloc(memVec);

  clock_t cpu_start0, cpu_end0;
  clock_t cpu_start1, cpu_end1;
  clock_t cpu_start2, cpu_end2;

  printf("\nCreating Matrix and Vector\n");
  cpu_start0 = clock();
  CreatMatVec(a, c, d, row, column);
  cpu_end0 = clock();

  printf("\nTime spent in creating vector and matrix   : %4.6f sec\n", (double)((double)(cpu_end0 - cpu_start0)/CLOCKS_PER_SEC));

  for(int i=0; i<column; i++)
  {
    d[i] = 0.0; 
    dBlas[i] = 0.0; 
  }

  printf("\n");
  printf("\nCalculating Matrix-Vector Multiplication using CPU function\n");
  cpu_start1 = clock();
  matVecMulti(a, c, d, row, column);
  cpu_end1 = clock();

  printf("\n");
  printf("\nTime spent in matrix vector multiplication : %4.6f sec\n", (double)((double)(cpu_end1 - cpu_start1)/CLOCKS_PER_SEC));

 // for(int i=0; i<column; i++)
 // {
 //  printf("d[%d] : %lf\n",i,d[i]);
 // }
 
  printf("\n");

  printf("\nCalculating Matrix-Vector Multiplication using BLAS\n");
  cpu_start2 = clock();
  cblas_dgemv(CblasRowMajor, CblasNoTrans, row, column, 1, a, column, c, 1, 1.0f, dBlas, 1);
  cpu_end2 = clock();

  printf("\nTime spent in matrix vector multiplication using blas : %4.6f sec\n", (double)((double)(cpu_end2 - cpu_start2)/CLOCKS_PER_SEC));
 
  //for(int i=0; i<column; i++)
  //{
  // printf("dBlas[%d] : %lf\n",i,dBlas[i]);
  //}

  for(int i=0; i<column; i++) 
  {
   diff +=  dBlas[i] - d[i];
  }
  if(diff == 0) 
    printf("\nResults are correct\n");  
  else 
    printf("\nResults are incorrect\n");  
  printf("\n");

  free(a);
  free(c);
  free(d);
  free(dBlas);
  return 0;	
}

