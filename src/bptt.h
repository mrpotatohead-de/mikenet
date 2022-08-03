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

#ifndef BPTT_H
#define BPTT_H

int  bptt_forward(Net *net,Example *ex);
Real bptt_unit_activation(Group *group,int unit,int tick);
Real bptt_value_activation(Group *g,Real val);

int  bptt_compute_gradients(Net *net,Example *ex);
Real bptt_unit_derivative(Group *group,int unit,int tick);
Real bptt_value_derivative(Group *g,Real output);
void bptt_compute_dedx(Net *net,Group *gto,Example *ex,int t,int i);
void bptt_backprop_error(Net *net,Connections *c,int t);
int  bptt_apply_deltas(Net *net);
int  bptt_apply_deltas_dbd(Net *net);
int  bptt_apply_deltas_dbd_momentum(Net *net);
int  bptt_apply_deltas_momentum(Net *net);
int  bptt_apply_deltas_decay(Net *net,float decay);
int  bptt_apply_deltas_store(Net *net);

Real clip_output(Group *g,Real v);
Real invert_activation(Group *g,Real o);
Real max_activation(Group *g);
Real min_activation(Group *g);
Real bounded_clip_output(Group *g,Real o,Real bound);
Real gradient_slope(Net *net);

#endif
