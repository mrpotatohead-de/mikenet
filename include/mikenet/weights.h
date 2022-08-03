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


/* for backwards compatibility */
#define load_weights(a,b) load_state(a,b)
#define save_weights(a,b) save_state(a,b)

int load_state(Net *net,char *fn);
int save_state(Net *net,char *fn);

int save_binary_weights(Net *net,char *fn);
int load_binary_weights(Net *net,char *fn);

void freeze_weights(Connections *c);
void unfreeze_weights(Connections *c);

void restore_weights(Connections *c);
void store_weights(Connections *c);
void restore_all_weights(Net *net);
void store_all_weights(Net *net);
void decay_all_weights(Net *net,Real v);
void decay_weights(Connections *c,Real v);
void init_dbdWeight(Connections *c);
void init_prevDeltas(Connections *c);

extern int compress_weight_file;
int num_weights_in_net(Net *net);

void weigand_decay_all_weights(Net *net,Real v);
void weigand_decay_weights(Connections *c,Real v);
