/*
    mikenet - a simple, fast, portable neural network simulator.
    Copyright (C) 1995  Michael W. Harm

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

    See file COPYING for a copy of the GNU General Public License.

    For more info, contact:

    Michael Harm
    HNB 126
    University of Southern California
    Los Angeles, CA 90089-2520

    email:  mharm@gizmo.usc.edu

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef USE_BLAS
#include <cblas.h>
#endif
#ifdef USE_AMDBLIS
#include "blis.h"
#endif

#include "const.h"
#include "matrix.h"

int default_useBlasThreshold = 100000;

#ifdef USE_BLAS
int default_useBlas = 1;
#else
int default_useBlas = 0;
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

void mikenet_matrix_vec_mult_p(Real *outvec, int nout, Real *invec,
                               int nin, Real **mat)
{
  int i, j;

  for (i = 0; i < nout; i++)
  {
#ifdef USE_OPENMP
omp_set_dynamic(0);     // Explicitly disable dynamic teams
omp_set_num_threads(2); // Use 2 threads for all consecutive parallel regions
#pragma omp parallel for
#endif
    for (j = 0; j < nin; j++)
    {
      outvec[i] += mat[i][j] * invec[j];
    }
  }
}

void mikenet_matrix_vec_mult_t_p(Real *outvec, int nout, Real *invec,
                                 int nin, Real **mat)
{
  int i, j;

  for (i = 0; i < nout; i++)
  {
#ifdef USE_OPENMP
omp_set_dynamic(0);     // Explicitly disable dynamic teams
omp_set_num_threads(2); // Use 2 threads for all consecutive parallel regions
#pragma omp parallel for
#endif
    for (j = 0; j < nin; j++)
    {
      outvec[i] += mat[j][i] * invec[j];
    }
  }
}

void mikenet_matrix_vec_mult(Real *outvec, int nout, Real *invec,
                             int nin, Real **mat)
{

  //printf("nout: %d, nin: %d\n",nout,nin);
  //printf("mat: %.1f %.1f %.1f %.1f\n",mat[0][0],mat[0][1],mat[0][2],mat[0][3]);
  //printf("x: %.1f %.1f %.1f %.1f\n",invec[0],invec[1],invec[2],invec[3]);
  //printf("y_in: %.1f %.1f %.1f %.1f\n",outvec[0],outvec[1],outvec[2],outvec[3]);
#ifdef USE_BLAS
  /* if it's too darn small, don't bother with blas */
  if (nout * nin < default_useBlasThreshold || default_useBlas == 0)
    mikenet_matrix_vec_mult_p(outvec, nout, invec, nin, mat);
  else
    cblas_sgemv(CblasRowMajor, CblasNoTrans, nout, nin, 1.0,
                (float *)(&mat[0][0]), nin, invec, 1, 1.0, outvec, 1);
#elif USE_AMDBLIS
  float alpha = 1.0, beta = 1.0;
  bli_sgemv( BLIS_NO_TRANSPOSE, BLIS_NO_CONJUGATE,nout, nin, &alpha, (float *)(&mat[0][0]), nin, 1, invec, 1, &beta, outvec, 1);
  //mikenet_matrix_vec_mult_p(outvec, nout, invec, nin, mat);
#else
  mikenet_matrix_vec_mult_p(outvec, nout, invec, nin, mat);
#endif
  //printf("y_out: %.1f %.1f %.1f %.1f\n",outvec[0],outvec[1],outvec[2],outvec[3]);
  //printf("y_out: %.1f %.1f %.1f %.1f\n",outvec[4],outvec[5],outvec[6],outvec[7]);
  
  
}

void mikenet_matrix_vec_mult_t(Real *outvec, int nout, Real *invec,
                               int nin, Real **mat)
{

#ifdef USE_BLAS
  if (nin * nout < default_useBlasThreshold || default_useBlas == 0)
    mikenet_matrix_vec_mult_t_p(outvec, nout, invec, nin, mat);
  else
    cblas_sgemv(CblasRowMajor, CblasTrans, nin, nout, 1.0,
                (float *)(&mat[0][0]), nout, invec, 1, 1.0, outvec, 1);
#elif USE_AMDBLIS
  float alpha = 1.0, beta = 1.0;
  bli_sgemv( BLIS_TRANSPOSE, BLIS_NO_CONJUGATE,nin, nout, &alpha, (float *)(&mat[0][0]), nout, 1, invec, 1, &beta, outvec, 1);
  //mikenet_matrix_vec_mult_t_p(outvec, nout, invec, nin, mat);
#else
  mikenet_matrix_vec_mult_t_p(outvec, nout, invec, nin, mat);

#endif
}

/* ok: this  does Y = A * B(t) + Y (the 't' means transpose)
   As a brush up: if A is dimension 2, and B is dimension 3,
   then Y better be dimension 2(rows) by 3(cols).
   Here, v1 is the A, and v2 is the B
*/

void mikenet_matrix_outer_product(Real **matrix,
                                  Real *v1,
                                  int n1, /* rows */
                                  Real *v2,
                                  int n2) /* cols */
{
  int i, j;

  if (default_useBlas)
  {
#ifdef USE_BLAS
    cblas_sger(CblasRowMajor, n1, n2, 1.0, v1, 1, v2, 1, (float *)(&matrix[0][0]),
               n2);
#elif USE_AMDBLIS
  float alpha = 1.0;
  bli_sger(BLIS_NO_CONJUGATE,BLIS_NO_CONJUGATE,n2,n1,&alpha,v2,1,v1,1,(float *)(&matrix[0][0]),n1,1);

#endif
    return;
  }
  else
  {
    for (i = 0; i < n1; i++)
    {
      for (j = 0; j < n2; j++)
      {
        matrix[i][j] += v1[i] * v2[j];
      }
    }
  }
}
