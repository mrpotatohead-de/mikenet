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

#ifndef NET_H
#define NET_H

#include "groupstruct.h"

#include "connstruct.h"

#include "netstruct.h"


Group *find_group_by_name(char *name);
Group * init_group(char *name,int numUnits,int time);



Group * init_bias(float val,int time); 
Connections * connect_groups(Group *gfrom, Group *gto);

Net * create_net();
Net * init_net();


extern int defaultGroupMethod(void *group,void *ex,int t);
extern int defaultConnectionMethod(void *c,void *ex,int t);
extern int defaultNetMethod(void *n,void *ex,int t);

void bind_connection_to_net(Net *net,Connections *c);
void unbind_connection_from_net(Net *net,Connections *c);
void bind_group_to_net(Net *net,Group *g);
void unbind_group_from_net(Net *net,Group *g);

void free_net(Net *net);
void free_group(Group *group);
void free_connections(Connections *c);

int is_group_in_net(Group *g,Net *net);

int name_units(Group *g,char *fn);
void randomize_connections(Connections *c,Real weightRange);
void precompute_topology(Net *net,Group *input);
void precompute_topology2(Net *net,Group *input,Group *input2);


extern Real default_weightDecay;
extern int default_noisyUpdate;
extern Real default_epsilon;
extern Real default_tao;
extern Real default_temperature;
extern int default_resetActivation;
extern int default_scaling;
extern Real default_errorRadius;
extern Real default_taoEpsilon;
extern Real default_weightNoise;
extern int default_activationType;
extern int default_errorComputation;
extern int globalNumGroups;  /* global number of groups defined */
extern Group **groups; /* global group list */
extern Real default_activationNoise;  
extern Real default_inputNoise;  
extern Real default_primeOffset;
extern Real default_momentum;
extern Real default_dbdUp,default_dbdDown;
extern int default_weightNoiseType;
extern Real default_weightNoise;
extern int default_tai;  /* by default, do we time average inputs? */
extern Real default_temporg;
extern Real default_tempmult;
extern int default_errorRamp;
extern Real default_taoDecay;
extern Real default_taoMaxMultiplier;
extern Real default_taoMinMultiplier;
#endif
