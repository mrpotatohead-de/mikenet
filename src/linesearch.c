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
#include "const.h"
#include "net.h"
#include "tools.h"
#include "weights.h"
#include "linesearch.h"

int
zero_gradients(Net *net)
{
  Real *d;
  Connections *c;
  int nc,i,j,x;

  nc=net->numConnections;
  for(x=0;x<nc;x++)
    {
      c=net->connections[x];
      for(i=0;i<c->to->numUnits;i++)
	{
	  d=c->gradients[i];
	  for(j=0;j<c->from->numUnits;j++)
	    {
	      *d++ =0;
	    }
	}
    }
  return 0;
}


/*******************************************************************/
/* presumes weights have been stored away by calling store_weights */
/*    takes weights in store_weights, applies current graident and */
/*    makes the result a new set of weights                        */
/*******************************************************************/
int
test_step(Net *net,Real epsilon)
{
  Real *w,*xi,*fromW;
  Connections *c;
  unsigned char *f;
  int nc,i,j,x,nfrom,nto;

  nc=net->numConnections;
  
  for(x=0;x<nc;x++)
    {
      c=net->connections[x];
      if (c->locked)
	continue;

      if (c->backupWeights==NULL)
	{
	  Error0("test_step: Weights not stored (call store_weights first)\n");
	  return 0;
	}
      nfrom=c->from->numUnits;
      nto=c->to->numUnits;
      for(i=0;i<nto;i++)
	{
	  fromW=c->backupWeights[i];
	  w=c->weights[i];
	  f=c->frozen[i];
	  xi=c->gradients[i];
	  for(j=0;j<nfrom;j++)
	    {
	      if (!f[j])
		{
		  w[j] = fromW[j] - (xi[j] * epsilon);
		}
	      else
		w[j] = fromW[j];
	    }
	}
    }
  return 0;
}


void 
init_cg(Net *net)
{
  int x,nfrom,nto,i,j,nc;
  Real *h,*g,*xi;
  Connections *c;

  nc=net->numConnections;

  for(x=0;x<nc;x++)
    {
      c=net->connections[x];
      if (c->locked)
	continue;

      if (c->h ==NULL)
	{
	  c->h=make_real_array(c->to->numUnits,c->from->numUnits);
	  for(i=0;i<c->to->numUnits;i++)
	    for(j=0;j<c->from->numUnits;j++)
	      c->h[i][j]=0.0;
	}
      if (c->g ==NULL)
	{
	  c->g=make_real_array(c->to->numUnits,c->from->numUnits);
	  for(i=0;i<c->to->numUnits;i++)
	    for(j=0;j<c->from->numUnits;j++)
	      c->g[i][j]=0.0;
	}
      nfrom=c->from->numUnits;
      nto=c->to->numUnits;
      for(i=0;i<nto;i++)
	{
	  xi=c->gradients[i];
	  g=c->g[i];
	  h=c->h[i];
	  for(j=0;j<nfrom;j++)
	    {
	      g[j] = -xi[j];
	      h[j] = g[j];
	      xi[j] = -h[j];
	    }
	}
    }
}


/* assumes init_cg was called once, to populate values of
   h and g arrays */
void
cg(Net *net)
{
  Real *g,*xi,dgg,gg,gam,*h;
  Connections *c;
  int nc,i,j,x,nfrom,nto;
  static int first=1;

  nc=net->numConnections;

  if (first)
    {
      first =0;
      for(x=0;x<nc;x++)
	{
	  c=net->connections[x];
	  if (c->locked)
	    continue;

	  nfrom=c->from->numUnits;
	  nto=c->to->numUnits;
	  for(i=0;i<nto;i++)
	    {
	      xi=c->gradients[i];
	      g=c->g[i];
	      h=c->h[i];
	      for(j=0;j<nfrom;j++)
		{
		  g[j] = -xi[j];
		  h[j] = g[j];
		  xi[j] = -h[j];
		}
	    }
	}
      return;
    }
  
  dgg=0.0;
  gg=0.0;
  for(x=0;x<nc;x++)
    {
      c=net->connections[x];
      if (c->locked)
	continue;

      nfrom=c->from->numUnits;
      nto=c->to->numUnits;
      for(i=0;i<nto;i++)
	{
	  xi=c->gradients[i];
	  g=c->g[i];
	  h=c->h[i];
	  for(j=0;j<nfrom;j++)
	    {
	      gg += g[j] * g[j];
	      dgg += (xi[j] + g[j]) * xi[j]; 
	    }
	}
    }
  /* now we've computed our gg and dgg */
  gam = dgg/gg;
  for(x=0;x<nc;x++)
    {
      c=net->connections[x];
      if (c->locked)
	continue;

      nfrom=c->from->numUnits;
      nto=c->to->numUnits;
      for(i=0;i<nto;i++)
	{
	  xi=c->gradients[i];
	  h=c->h[i];
	  g=c->g[i];
	  for(j=0;j<nfrom;j++)
	    {
	      g[j] = -xi[j];
	      h[j] = g[j] + gam * h[j];
	      xi[j] = -h[j];
	    }
	}
    }
}







