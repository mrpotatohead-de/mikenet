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

/* formula for running variance:

   (sumsquared - mean*sum)/(n-1)

   where sumsquared = sum of x^2
   sum is sum of x
   mean is sum/n

   sd = sqrt(variance);

   standard error = standard deviation/(sqrt(n))
   where n is number of samples

*/

Real tanh_prime();

#define init_matrix(r1,r2,c1,c2,matrix,value) \
   { \
      int i,j; \
      for(i=r1;i<r2;i++)  \
        for(j=c1;j<c2;j++) \
          matrix[i][j]=value; \
   } 


FILE * mikenet_open_for_reading(char *fn,char *newfn,int *is_tmpfile);

void ** make_array(int row,int col,int size);

void free_array(void **p);
void free_real_array(Real **p);

Real **make_real_array(int row,int col);

double linear_activation(Real x, Real temp);
double step_activation(Real x, Real temp);

double sigmoid_activation(Real x, Real temp);
double sigmoid_derivative(Real x,Real temp);
double sigmoid_inverse(double y,double temp);
double tanh_inverse(double y,double temp);

double fast_sigmoid_activation(Real x, Real temp);


double tanh_activation();
double tanh_derivative();


/* base 2 log */
double log_2(double x);

Real gen_random_weight(Real range);

extern char default_compressor[255];
extern char default_decompressor[255];

void announce_version();

void *mh_malloc(int size);
void *mh_realloc(void *p,int size);
void *mh_calloc(int size1,int size2);

#ifdef _CRAY
double atanh(double x);
#endif

void Error0();
void Error1();
void Error2();
void Error3();
void Choke0();

#define TANH_MIN -0.999999
#define TANH_MAX 0.999999
#define LOGISTIC_MIN 0.000001
#define LOGISTIC_MAX 0.999999

Real CLIP(Real x,Real mn,Real mx);
