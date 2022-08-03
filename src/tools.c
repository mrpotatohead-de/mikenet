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
#include <errno.h>

#ifdef unix
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#endif


#include "const.h"
#include "net.h"
#include "example.h"
#include "tools.h"
#include "random.h"
#include "fastexp.h"

#ifdef linux
char default_compressor[255]="gzip";
char default_decompressor[255]="gunzip";
#else
char default_compressor[255]="compress";
char default_decompressor[255]="uncompress";
#endif



#ifdef _CRAY
/* stupid cray math library doesn't give you atanh... */
double atanh(double x)
{
  double y;
  y=0.5 * log((1.0+x)/(1.0-x)); /* CRC, 29th ed, pg 163 */
  return y;
}
#endif

Real CLIP(Real x,Real mn,Real mx)
{
  if (x > mx) return mx;
  else if (x < mn) return mn;
  else return x;
}

FILE *
mikenet_open_for_reading(char *fn,char *newfn,int *is_tmpfile)
{
  FILE *f=NULL;
  char line[255],cmd[255],*p;
  int seed,rc;
  
  strcpy(newfn,"");
  *is_tmpfile=0;
  

#ifdef unix
  /* test if its a .Z file */
  if (strlen(fn) > 2)
    {
      p=&fn[strlen(fn)-2];
      if (strcmp(p,".Z")==0)
	{
	  /* if so, uncompress to new file */
	  seed=((int)(mikenet_random() * 10000.0)) % 10000;
	  sprintf(newfn,"TMP_%d_%d",(int)getpid(),seed);
	  sprintf(cmd,"zcat %s > %s",fn,newfn);
	  system(cmd);
	  f=fopen(newfn,"r");
	  if (f==NULL)
	    {
	      Error1("Can't open file %s",fn);
	      return NULL;
	    }
	  *is_tmpfile = 1;
	  return f;
	}
    }
  /* test if its a .gz file */
  if (strlen(fn) > 3)
    {
      p=&fn[strlen(fn)-3];
      if (strcmp(p,".gz")==0)
	{
	  /* if so, uncompress to new file */
	  seed=((int)(mikenet_random() * 10000.0)) % 10000;
	  sprintf(newfn,"TMP_%d_%d",(int)getpid(),seed);
	  sprintf(cmd,"zcat %s > %s",fn,newfn);
	  system(cmd);
	  f=fopen(newfn,"r");
	  if (f==NULL)
	    {
	      Error1("Can't open file %s",fn);
	      return NULL;
	    }
	  *is_tmpfile = 1;
	  return f;
	}
    }

  /* normal file as given */

  f=fopen(fn,"r");
  /* if we fail to open, see if its compressed */
  if (f==NULL)
    {
      strcpy(line,fn);
      strcat(line,".Z");
      f=fopen(line,"r");
      /* does compressed file exist? */
      if (f != NULL)
	{
	  /* found .Z file */
	  fclose(f); 
	  /* if so, uncompress to new file */
	  seed=((int)(mikenet_random() * 10000.0)) % 10000;
	  sprintf(newfn,"TMP_%d_%d",(int)getpid(),seed);
	  sprintf(cmd,"zcat %s > %s",fn,newfn);
	  system(cmd);
	  f=fopen(newfn,"r");
	  if (f==NULL)
	    {
	      Error1("Can't open file %s",fn);
	      return NULL;
	    }
	  *is_tmpfile = 1;
	  return f;
	}
      else /* can't open .Z file, try .gz */
	{
	  strcpy(line,fn);
	  strcat(line,".gz");
	  f=fopen(line,"r");
	  if (f==NULL)
	    {
	      Error1("Can't open file %s (not .Z or .gz)",fn);
	      return NULL;
	    }
	  /* gzipped; so open it */
	  fclose(f);
	  /* if so, uncompress to new file */
	  seed=((int)(mikenet_random() * 10000.0)) % 10000;
	  sprintf(newfn,"TMP_%d_%d",(int)getpid(),seed);
	  sprintf(cmd,"%s -c %s > %s ",default_decompressor,fn,newfn);
	  rc=system(cmd);
	  f=fopen(newfn,"r");
	  /* now try it */
	  if (f==NULL)
	    {
	      Error2("Can't open decompressed file %s from %s",fn,newfn);
	      return NULL;
	    }
	  *is_tmpfile = 1;
	  return f;	  
	}
    }
#else
  /* not unix; die if file not opened correctly */
  f=fopen(fn,"r");
  if (f==NULL)
    {
      Error1("Can't open file %s",fn);
      return NULL;
    }
#endif
  if (f==NULL) 
    Error1("Can't open file %s\n",fn);
  return f;
}

void * mh_malloc(size)
int size;
{
  void *p;
  p=(void *)malloc(size);
  if (p==NULL) 
    Choke0("Not enough Memory. aborting.  sorry.");
  return p;
}

double log_2(double x)
{
  double y;
  y=log(x) / log((double)2.0);
  return y;
}

void * mh_calloc(size,size2)
int size,size2;
{
  void *p;
  p=(void *)calloc(size,size2);
  if (p==NULL) 
    Choke0("Not enough Memory. aborting.  sorry.");
  return p;
}

void * mh_realloc(ptr,size)
void *ptr;     
int size;
{
  void *p;
  p=(void *)realloc(ptr,size);
  if (p==NULL) 
    Choke0("Not enough Memory. aborting.  sorry.");
  return p;
}

void free_array(void **v)
{
  void *p;
  p = v[0];
  free(p);
  free(v);
}

void free_real_array(Real **p)
{
  free_array((void **)p);
}

void ** make_array(row,col,size)
int row, col, size;
{
  void **vec;
  char *tmp,*p;
  int i;
  vec=(void **)mh_malloc(row*sizeof(void *));
  if (vec==NULL)
    Choke0("out of memory error!");
  tmp=(char *)mh_malloc(row * col * size);
  if (tmp==NULL)
    Choke0("out of memory error!");
  p=tmp;
  for(i=0;i<row;i++)
    {
      vec[i]=(void *)p;
      p+= col*size;
    }
  return vec;
}

Real **make_real_array(row,col)
int row, col;
{
  Real **v;
  int i,j;
  v=(Real **)make_array(row,col,sizeof(Real));
  for(i=0;i<row;i++)
    for(j=0;j<col;j++)
      v[i][j]=0.0;
  return v;
}

Real
gen_random_weight(Real range)
{
  Real r,val;
  r=mikenet_random();
  val=r*2*range - range;
  return val;
}

double square(x)
double x;
{
  return x*x;
}


double tanh_activation(x,temp)
Real x,temp;
{
  double v;
  v=x * temp;
  if (v > 10.0)
    return 1.0;
  else if (v < -10.0)
    return -1.0;
  else return tanh((double)v);
}

double tanh_derivative(y,temp)
Real y,temp;
{
  double v;
  v=y * temp;
  if (v > 0.99999)
    return 0.0;
  else if (v < -0.99999)
    return 0.0;
  else return (temp * (1.0 - y * y));
}

double sigmoid_activation(Real x,Real temp)
{
  double v,y;
  v = (double)temp * (double)x;

  if (v > 15.0)
    return LOGISTIC_MAX;
  else if (v < -15.0)
    return LOGISTIC_MIN;
  else 
    {
      y = 1.0/(1.0 + exp(((double)(-1.0)) * v));
      return CLIP(y,LOGISTIC_MIN,LOGISTIC_MAX);
    }
}

double linear_activation(Real x,Real temp)
{
  double v;
  v = (double)temp * (double)x;

  return v;
}

double step_activation(Real x,Real temp)
{
  double v;
  v = (double)temp * (double)x;
  if (v > 0.0)
    return 1.0;
  else return 0.0;
}



double sigmoid_derivative(Real y,Real temp)
{
  double v;
  v = (double)temp * (double)y;

  v=CLIP(v,LOGISTIC_MIN,LOGISTIC_MAX);
  return ((double)temp)*((double)y * (1.0-(double)y));
}

double fast_sigmoid_activation(Real x,Real temp)
{
  double v,y;

  v = (double)temp * (double)x * -1.0;

  y=1.0/(1.0 + myexp(v));

  return CLIP(y,LOGISTIC_MIN,LOGISTIC_MAX);
}




/* given y, and a temperature, what produces that output? */
double sigmoid_inverse(double y,double temp)
{
  double v;

  if (y > LOGISTIC_MAX)
    return 15.0;
  else if (y < LOGISTIC_MIN)
    return -15.0;
  v= -(log((1.0/y)-1))  / temp;
  return v;
}

double tanh_inverse(double y,double temp)
{
  double v;
  v = atanh(y)/temp;
  return v;
}

/* give it a pointer.  it stuffs a line into the array
   pointed to by 'out' and returns pointer to where it
   left off */
char *
eat_line(buf,out)
char *buf;
char *out;
{
  char *p;
  p=buf;
  while(*p && *p != '\n')
    {
      *out++=*p++;
    }
  *out=0;
  if (*p=='\n') p++;
  return p;
}





#ifdef unix
#ifndef linux
#ifndef HPC89
#ifndef CONVEX
int matherr(ex)
struct exception *ex;
{
  fprintf(stderr,"MATH EXCEPTION: type %d name %s arg %f\n",
	  ex->type,ex->name,ex->arg1); 
  exit(1);
  return -1;  /* so compilers don't whine about no return value */
}
#endif
#endif
#endif
#endif  /* unix */





void
Error0(fmt)
char *fmt;
{
  fprintf(stderr,"Error: ");
  fprintf(stderr,fmt);
  fprintf(stderr,"\n");
  exit(-1);
}

void
Error1(fmt,arg)
char *fmt;
void *arg;
{
  fprintf(stderr,"Error: ");
  fprintf(stderr,fmt,arg);
  fprintf(stderr,"\n");
  exit(-1);
}

void
Error2(fmt,arg1,arg2)
char *fmt;
void *arg1,*arg2;
{
  fprintf(stderr,"Error: ");
  fprintf(stderr,fmt,arg1,arg2);
  fprintf(stderr,"\n");
  exit(-1);
}

void
Error3(fmt,arg1,arg2,arg3)
char *fmt;
void *arg1,*arg2,*arg3;
{
  fprintf(stderr,"Error: ");
  fprintf(stderr,fmt,arg1,arg2,arg3);
  fprintf(stderr,"\n");
  exit(-1);
}

void
Choke0(fmt)
char *fmt;
{
  fprintf(stderr,"Error: ");
  fprintf(stderr,fmt);
  fprintf(stderr,"\n");
  exit(1);
}

void announce_version()
{
  printf("\n\n");
  printf("   Mikenet Version %d.%d.%d%c %s  Copyright (C) 1995 Michael W. Harm\n",
	 MIKENET_MAJOR_VERSION,
	 MIKENET_MINOR_VERSION,
	 MIKENET_UPDATE_NUMBER,
	 MIKENET_BUG_FIX_NUMBER,
	 MIKENET_VERSION_TAG);
  printf("   Mikenet comes with ABSOLUTELY NO WARRANTY \n");
  printf("   This is free software, and you are welcome to redistribute it\n");
  printf("     under certain conditions.\n"); 
  if (sizeof(Real)==4)
    printf("   using single-precision floats\n");
  else if (sizeof(Real)==8)
    printf("   using double-precision floats\n");
  else printf("  using %d byte floats\n",(int)sizeof(Real));
#ifdef USE_BLAS
  printf("   Compiled with BLAS matrix library\n");
#endif
#ifdef USE_MPI
  printf("   Compiled with MPI parallel processing library\n");
#endif
  printf("   Compiled using %s on machine type %s\n",COMPILER,MACHINE);

#ifdef __TIME__
  printf("   Compiled at time %s\n",__TIME__);
#endif
#ifdef __DATE__
  printf("   Compiled on date %s\n",__DATE__);
#endif
  printf("\n");
}
