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
#include "analyze.h"

void sever_probabilistically(Connections *c,float p)
{
  Group *to,*from;
  float dice;
  int i,j;

  to=c->to;
  from=c->from;
  for(i=0;i<to->numUnits;i++)
    for(j=0;j<from->numUnits;j++)
      {
	dice=mikenet_random();
	if (dice <=p)
	  {
	    c->weights[i][j]=0;
	    c->frozen[i][j]=1;
	  }
      }
}  

void sever(Connections *c)
{
  sever_probabilistically(c,1.001);
}


Real contribution_from_unit(Connections *c,int tounit,int fromunit,int time)
{
  return (c->from->outputs[time-1][fromunit] *
	  c->weights[tounit][fromunit]);
  
}

Real contribution_from_group(Connections *c,int tounit,int time)
{
  Real sum=0;
  int i;

  for(i=0;i<c->from->numUnits;i++)
    sum += contribution_from_unit(c,tounit,i,time);
  return sum;
}


