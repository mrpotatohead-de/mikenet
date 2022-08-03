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
#include "random.h"

/* how many gaussians do we make (use prime number) */
#define GAUSSIAN_MAX 100003

#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define MASK 123459876


static long idnum=666; /* random number of the beast */

void mikenet_set_seed(i)
long i;     
{
  idnum=i;
}

long mikenet_get_seed()
{
  return idnum;
}

float mikenet_random()
{
  long k;
  float ans;
  idnum ^= MASK;
  k=idnum/IQ;
  idnum=IA*(idnum-k*IQ)-IR*k;
  if (idnum < 0)
    idnum +=IM;
  ans=AM*idnum;
  idnum ^= MASK;
  return ans;
}



static int gaussian_count=0;
static float *gaussians=NULL;

float get_gaussian()
{
  if (gaussians==NULL)
    init_gaussians();
  gaussian_count++;
  if (gaussian_count >= GAUSSIAN_MAX)
    gaussian_count=0;
  return gaussians[gaussian_count];
}

float gaussian_number()
{
  static int iset=0;
  static float gset;
  float fac,rsq,v1,v2;
  if (iset==0)
    {
      do {
	v1=2.0*mikenet_random()-1.0;
	v2=2.0*mikenet_random()-1.0;
	rsq=v1*v1+v2*v2;
      }
      while(rsq >= 1.0 || rsq==0.0);
      fac=sqrt(-2.0*log(rsq)/rsq);
      gset=v1*fac;
      iset=1;
      return v2*fac;
    }
  else
    {
      iset=0;
      return gset;
    }
}

void init_gaussians()
{
  int i;
  if (gaussians==NULL)
    {
      gaussians=(float *)calloc(sizeof(float),GAUSSIAN_MAX);
      for(i=0;i<GAUSSIAN_MAX;i++)
	gaussians[i]=gaussian_number();
    }
}

void set_gaussian_seed(int seed)
{
  gaussian_count=seed % GAUSSIAN_MAX;
}
  
