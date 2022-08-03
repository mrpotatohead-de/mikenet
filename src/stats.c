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
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "stats.h"

#define setNaN(x) { float _d1 =0.0; x = 1.0/_d1 - 2.0/_d1; }

#define INIT_STAT_ARRAY_SIZE 1000

StatStruct *
get_stat_struct()
{
  StatStruct *s;

  s=malloc(sizeof(StatStruct));
  if (s==NULL)
    {
      fprintf(stderr,"get_stat_struct: error allocating stat structure\n");
      exit(-1);
    }
  s->n=0;
  s->array_size=INIT_STAT_ARRAY_SIZE;
  s->array=calloc(sizeof(float),INIT_STAT_ARRAY_SIZE);
  if (s->array==NULL)
    {
      fprintf(stderr,"get_stat_struct: error allocating dataarray\n");
      exit(-1);
    }
  
  s->sum=0;
  s->sum_of_squares=0;
  return s;
}

void
push_item(float v,StatStruct *s)
{
  /* if array full, float size */
  if (s->n == s->array_size)
    {
      s->array_size *= 2;
      s->array=realloc(s->array,
		       s->array_size * sizeof(float));
      if (s->array==NULL)
	{
	  fprintf(stderr,"push_item: error allocating larger array\n");
	  exit(-1);
	}
    }
 
  if (s->n==0)
    s->min = s->max = v;
  else
    {
      if (v > s->max)
	s->max=v;
      if (v < s->min)
	s->min=v;
    }
    
  s->array[s->n] = v;
  s->sum += v;
  s->sum_of_squares += v * v;
  s->n++;
}

void
clear_stats(StatStruct *s)
{
  s->n=0;
  s->sum=0;
  s->sum_of_squares=0;
}

void free_stats(StatStruct *s)
{
  free(s->array);
  free(s);
}

int my_float_compare(const void *xv,const void *yv)
{
  float *x,*y;
  x=(float *)xv;
  y=(float *)yv;
  if (*x == *y)
    return 0;
  else if (*x < *y)
    return -1;
  else return 1;
}

int my_float_compare_descending(const void *xv,const void *yv)
{
  float *x,*y;
  x=(float *)xv;
  y=(float *)yv;
  if (*x == *y)
    return 0;
  else if (*x < *y)
    return 1;
  else return -1;
}

void sort_stats(StatStruct *s)
{
  qsort((void *)s->array,s->n,sizeof(float),my_float_compare);
}

void sort_stats_descending(StatStruct *s)
{
  qsort((void *)s->array,s->n,sizeof(float),my_float_compare_descending);
}


float
median(StatStruct *s)
{
  float v;
  int n=s->n,i;
  sort_stats(s);
  if ((n & 0x01) == 0) /* even number, take average */
    {
      i=(n/2)-1;
      if (i<0)
	i=0;
      if (i==n-1)
	v=s->array[i];
      else
	v=(s->array[i] + s->array[1+i])/2.0;
    }
  else 
    v=s->array[(int)(n/2)];
  return v;
}
  

float 
mean(StatStruct *s)
{
  if (s->n<1)
    {
      fprintf(stderr,"attempt to take mean of dataset with < 1 items\n");
      return 0.0;
    }
  return (s->sum / (float)s->n);
}


float variance(StatStruct *s)
{
  float mean,n,var;
  n=s->n;
  if (n<=1)
    {
      fprintf(stderr,"Error: Attempt to take standard deviation from dataset with < 2 items\n");
      return 0.0;
    }
  
  mean=s->sum / n;
  
  var = (s->sum_of_squares - mean*s->sum)/(n-1);
  /* the only way var is < 0 is machine precision
     roundoff error */
  if (var<0.0)
    var=0.0;
  return var;
}

float sd(StatStruct *s)
{
  float var;
  var=variance(s);
  return sqrt(var);
}

float se(StatStruct *s)
{
  float std=0.0,n;
  n=s->n;
  if (n<=1)
    {
      fprintf(stderr,"Error: Attempt to take standard error from dataset with < 2 items\n");
      return 0.0;
    }
  
  std=sd(s);
  
  return std/(sqrt(n));
}

float pearson(StatStruct *x,StatStruct *y)
{
  if (x->n != y->n)
    {
      fprintf(stderr,"Pearson: arrays not equal size\n");
      exit(-1);
    }
  return pearson_raw(x->array,y->array,x->n);
}
  

float pearson_raw(float *x,float *y,int n)
{
  float syy=0,sxy=0,sxx=0,ay,ax,r;
  float xt,yt;
  int equal=1;
  
  int i;
  
  ay=0;
  ax=0;
  for(i=0;i<n;i++)
    {
      ax+= x[i];
      ay+= y[i];
      if (x[i] != y[i])
	equal=0;
    }

  if (equal)
    return 1.0;

  ax /= n;
  ay /= n;
    
  for(i=0;i<n;i++)
    {
      xt = x[i]-ax;
      yt = y[i]-ay;
      sxx += xt*xt;
      syy += yt*yt;
      sxy += xt*yt;
    }
  if (sxx==0 || syy==0)
    return 0.0;

  r=sxy/sqrt(sxx*syy);
  return r;
}
