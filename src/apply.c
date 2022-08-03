/*
    mikenet - a simple, fast, portable neural network simulator.
    Copyright (C) 1995  Michael W. Harm

    This program is free software you can redistribute it and/or modify
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
#include "weights.h"
#include "tools.h"


int
bptt_apply_deltas(net)
Net *net;
{
  Real *w,*g,e;
  unsigned char *f;
  Connections *c;
  int nc,i,j,x,nto,nfrom;

  nc=net->numConnections;
  for(x=0;x<nc;x++)
    {
      c=net->connections[x];
      if (c->locked)
	continue;
      nto=c->to->numUnits;
      e=c->epsilon;
      for(i=0;i<nto;i++)
	{
	  w=c->weights[i];
	  g=c->gradients[i];
	  f=c->frozen[i];
	  nfrom=c->numIncoming[i];
	  for(j=0;j<nfrom;j++)
	    {
	      if (*f++==0)
		{
		  *w -= *g  * e ;
		}
	      *g++ =0;
	      w++;
	    }
	}
    }
  return 0;
}

int
bptt_apply_deltas_decay(Net *net,float decay)
{
  Real *w,*g,e,d;
  unsigned char *f;
  Connections *c;
  int nc,i,j,x,nto,nfrom;

  d=1.0-decay;

  nc=net->numConnections;
  for(x=0;x<nc;x++)
    {
      c=net->connections[x];
      if (c->locked)
	continue;
      nto=c->to->numUnits;
      e=c->epsilon;
      for(i=0;i<nto;i++)
	{
	  w=c->weights[i];
	  g=c->gradients[i];
	  f=c->frozen[i];
	  nfrom=c->numIncoming[i];
	  for(j=0;j<nfrom;j++)
	    {
	      if (*f++==0)
		{
		  *w -= *g  * e ;
		  *w *= d;
		}
	      *g++ =0;
	      w++;
	    }
	}
    }
  return 0;
}

int
bptt_apply_deltas_dbd(net)
Net *net;
{
  Real *w,*g,e,*prev,*dbd,dx;
  unsigned char *f;
  Connections *c;
  int nc,i,j,x,nto,nfrom;

  nc=net->numConnections;
  for(x=0;x<nc;x++)
    {
      c=net->connections[x];
      if (c->locked)
	continue;
      nto=c->to->numUnits;
      if (c->dbdWeight==NULL)
	{
	  init_dbdWeight(c);
	}
      if (c->prevDeltas==NULL)
	{
	  init_prevDeltas(c);
	}
      e=c->epsilon;
      for(i=0;i<nto;i++)
	{
	  w=c->weights[i];
	  g=c->gradients[i];
	  f=c->frozen[i];
	  dbd = c->dbdWeight[i];
	  prev=c->prevDeltas[i];
	  nfrom=c->numIncoming[i];
	  for(j=0;j<nfrom;j++)
	    {
	      if (*f++==0)
		{
		  dx = (*g) * e * (*dbd);
		  *w -= dx;
		}
	      else dx=0;
	      if (*prev * *g < 0)
		*dbd *= c->dbdDown;
	      else *dbd += c->dbdUp;
	      dbd++;
	      *prev++ = dx;
	      *g++ =0;
	      w++;
	    }
	}
    }
  return 0;
}

int
bptt_apply_deltas_dbd_momentum(net)
Net *net;
{
  Real *w,*g,e,*dbd,*prevDeltas,m,del;
  unsigned char *f;
  Connections *c;
  int nc,i,j,x,nto,nfrom;

  nc=net->numConnections;
  for(x=0;x<nc;x++)
    {
      c=net->connections[x];
      if (c->locked)
	continue;

      nto=c->to->numUnits;

      if (c->dbdWeight==NULL)
	init_dbdWeight(c);

      if (c->prevDeltas==NULL)
	init_prevDeltas(c);

      e=c->epsilon;
      for(i=0;i<nto;i++)
	{
	  w=c->weights[i];
	  g=c->gradients[i];
	  f=c->frozen[i];
	  m=c->momentum;
	  prevDeltas=c->prevDeltas[i];
	  dbd = c->dbdWeight[i];
	  nfrom=c->numIncoming[i];
	  for(j=0;j<nfrom;j++)
	    {
	      if (*f++==0)
		{
		  del = ((*g) * e * (*dbd))
		    + (m * (*prevDeltas));
		  *w -= del;
		}
	      else del=0.0;
	      if (*prevDeltas * *g < 0)
		*dbd *= c->dbdDown;
	      else *dbd += c->dbdUp;
	      *prevDeltas++  = del;
	      dbd++;
	      *g++ =0;
	      w++;
	    }
	}
    }
  return 0;
}

int
bptt_apply_deltas_momentum(net)
Net *net;
{
  Real *w,*g,e,*prevDeltas,m,del;
  unsigned char *f;
  Connections *c;
  int nc,i,j,x,nto,nfrom;

  nc=net->numConnections;
  for(x=0;x<nc;x++)
    {
      c=net->connections[x];
      if (c->locked)
	continue;

      nto=c->to->numUnits;

      if (c->prevDeltas==NULL)
	init_prevDeltas(c);

      e=c->epsilon;
      for(i=0;i<nto;i++)
	{
	  w=c->weights[i];
	  g=c->gradients[i];
	  f=c->frozen[i];
	  m=c->momentum;
	  prevDeltas=c->prevDeltas[i];
	  nfrom=c->numIncoming[i];
	  for(j=0;j<nfrom;j++)
	    {
	      if (*f++==0)
		{
		  del = *g * e +
		    m * *prevDeltas;
		  *w -= del;
		}
	      else del=0.0;
	      *prevDeltas++ = del;
	      *g++ =0;
	      w++;
	    }
	}
    }
  return 0;
}




int bptt_apply_deltas_store(net)
Net *net;
{
  Real *w,*g,e,*prevDeltas,del;
  unsigned char *f;
  Connections *c;
  int nc,i,j,x,nto,nfrom;

  nc=net->numConnections;
  for(x=0;x<nc;x++)
    {
      c=net->connections[x];
      if (c->locked)
	continue;

      nto=c->to->numUnits;

      if (c->prevDeltas==NULL)
	init_prevDeltas(c);

      e=c->epsilon;
      for(i=0;i<nto;i++)
	{
	  w=c->weights[i];
	  g=c->gradients[i];
	  f=c->frozen[i];
	  prevDeltas=c->prevDeltas[i];
	  nfrom=c->numIncoming[i];
	  for(j=0;j<nfrom;j++)
	    {
	      if (*f++==0)
		{
		  del = *g * e;
		  *w -= del;
		}
	      else del=0.0;
	      *prevDeltas++ =del;
	      *g++ =0;
	      w++;
	    }
	}
    }
  return 0;
}


Real gradient_slope(Net *net)
{
  Connections *c;
  int i,j,k;
  double gmag=0,dmag=0;
  int n=0;
  double dot=0.0;

  for(i=0;i<net->numConnections;i++)
    {
      c=net->connections[i];
      if (c->prevDeltas !=NULL)
	{
	  for(k=0;k<c->to->numUnits;k++)
	    for(j=0;j<c->from->numUnits;j++)
	      {
		dot += c->prevDeltas[k][j] *
		  c->gradients[k][j];
		gmag += c->gradients[k][j] * c->gradients[k][j];
		dmag += c->prevDeltas[k][j] * c->prevDeltas[k][j];
		n++;
	      }
	}
    }
  if (n>0)
    return (Real)(dot / (sqrt(gmag) * sqrt(dmag)));
  else return 0.0;
}

