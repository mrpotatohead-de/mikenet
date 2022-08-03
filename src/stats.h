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

#ifndef MIKENET_STATS
#define MIKENET_STATS

typedef struct
{
  int n;  /* how many items */
  float *array;  /* data */
  float sum;
  float sum_of_squares;  /* sum of squared items */
  float min,max; /* min and max of items seen */
  int array_size;
} StatStruct;

/* allocate a statistics structure */
StatStruct * get_stat_struct();

/* add item v to statstruct */
void push_item(float v,StatStruct *s);

/* reset sums and counters (but doesn't free memory) */
void clear_stats(StatStruct *s);

/* free memory pointed to by s */
void free_stats(StatStruct *s);

float mean(StatStruct *s);
float median(StatStruct *s);

/* sort into ascending order */
void sort_stats(StatStruct *s);

/* sort into descending order */
void sort_stats_descending(StatStruct *s);

/* variance */
float variance(StatStruct *s);

/* standard dev */
float sd(StatStruct *s);

/* standard error */
float se(StatStruct *s);

/* pearson r value */
float pearson(StatStruct *x,StatStruct *y);

/* pearson, from simple arrays */
float pearson_raw(float *x,float *y,int n);

#endif
