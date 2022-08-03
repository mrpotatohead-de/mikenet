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

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "token.h"



char * 
get_line(p,out)
char *p,*out;
{
  while (*p && *p != '\n')
    {
      *out++=*p++;
    }
  *out=0;  /* force a null */

  if (*p) /* if we terminate because of a cr */
    p++;  /* hop over cr */
  
  return p;
}

char * 
get_token(p,out)
char *p,*out;
{
  while(*p && (*p=='\n' || *p==' ' || *p=='\t'))
    p++;
  
  while (*p && *p != '\n' && *p!=' ' && *p != '\t')
    {
      *out++=*p++;
    }
  *out=0;
  if (*p) /* we terminate because of a cr or white space */
    p++;
  
  return p;
}
