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


#ifndef EXAMPLE_H
#define EXAMPLE_H

typedef enum { CLAMP_HARD, CLAMP_SOFT} ExampleClampType;
typedef enum { SPARSE_EXAMPLE, EXPANDED_EXAMPLE} ExampleType;
typedef enum { CLAMP, TARGET} SpecificationType;

#include "ex_struct.h"

#include "ex_set_struct.h"

ExampleSet * load_examples(char *fn,int maxtime);
Example * get_sequential_example(ExampleSet *set);
Example * get_random_example(ExampleSet *set);
Example * create_example(int time);
void free_example(Example *ex);

void zap_example(Example *e);

int has_value(ExampleData *** v,int group,int time,int unit);

Real get_value(ExampleData *** v,int group,int time,int unit);

int find_sparse_value(SparseExample *ex, int unit);

extern Real default_exampleOnValue;
extern Real default_exampleOffValue;

void apply_example_clamps(Example *ex,Group *g,int t);
void apply_example_targets(Example *ex,Group *g,int t);

void default_ex_clamps_premethod(Example *ex,Group *g,int t);
void default_ex_clamps_postmethod(Example *ex,Group *g,int t);

extern void (*ex_clamps_premethod)(Example *ex,Group *g,int t);
extern void (*ex_clamps_postmethod)(Example *ex,Group *g,int t);

void default_ex_targets_premethod(Example *ex,Group *g,int t);
void default_ex_targets_postmethod(Example *ex,Group *g,int t);

extern void (*ex_targets_premethod)(Example *ex,Group *g,int t);
extern void (*ex_targets_postmethod)(Example *ex,Group *g,int t);

extern Real default_softClampThresh;

#endif




