/* to compile this, your MIKENET_DIR environment variable
   must be set to the appropriate directory.  putting this
   at the bottom of your .cshrc file will do the trick:

   setenv MIKENET_DIR ~mharm/Mikenet


   This demo program solves the xor problem.  It uses
   batch learning.

   To build the binary, type 'make xor'

   options:
       
            -epsilon #             (sets the learning rate to #)
	    -seed #                (sets random number seed to #)
	    -momentum #            (sets momentum parameter)
            -range #               (sets initial random weight range)

*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef unix
#include <unistd.h>
#endif

#include <mikenet/simulator.h>

/* number of time slices for the net */
#define TICKS 3

#define REP 10
#define ITER 1000

int main(int argc,char *argv[])
{
  Net *net;
  float momentum=0.0;
  Group *input,*hidden,*output,*bias;
  ExampleSet *examples;
  Example *ex;
  Connections *c1,*c2,*c3,*c4;
  int i,count,j;
  Real error;
  int seed=666;
  Real epsilon,range;

  announce_version();

  /* set the random number seed to some default */
  seed=getpid();

  /* a default learning rate */
  epsilon=0.5;
  
  /* default weight range */
  range=0.5;

  default_errorRadius=0.0;

  /* what are the command line arguments? */
  for(i=1;i<argc;i++)
    {
      if (strncmp(argv[i],"-epsilon",5)==0)
	{
	  epsilon=atof(argv[i+1]);
	  i++;
	}
      else if (strncmp(argv[i],"-momentum",5)==0)
	{
	  momentum=atof(argv[i+1]);
	  i++;
	}
      else if (strncmp(argv[i],"-seed",5)==0)
	{
	  seed=atoi(argv[i+1]);
	  i++;
	}
      else if (strcmp(argv[i],"-range")==0)
	{
	  range=atof(argv[i+1]);
	  i++;
	}
      else if (strcmp(argv[i],"-errorRadius")==0)
	{
	  default_errorRadius=atof(argv[i+1]);
	  i++;
	}
      else
	{
	  fprintf(stderr,"unknown argument: %s\n",argv[i]);
	  exit(-1);
	}
    }

  /* use cross entropy */
  default_errorComputation=CROSS_ENTROPY_ERROR;

  /* logistic activation */
  default_activationType=LOGISTIC_ACTIVATION;

  /* set the seed for the random number generator */
  mikenet_set_seed(seed);

  /* set the momentum to zero initially */
  default_momentum=0.0;
  
  /* build a network, with TICKS number of ticks */
  net=create_net(TICKS);
  
  /* learning rate: applies to connections being created  */
  default_epsilon=epsilon;

  /* create our groups.  
     format is: name, num of units, ticks */
  input=init_group("Input",2,TICKS);
  hidden=init_group("hidden",3,TICKS);
  output=init_group("Output",1,TICKS);

  /* bias is special.  format is: value, ticks */
  bias=init_bias(1.0,TICKS);   

  /* now add our groups to the network object */
  bind_group_to_net(net,input);
  bind_group_to_net(net,hidden);
  bind_group_to_net(net,output); 
  bind_group_to_net(net,bias); 

  /* now connect our groups, instantiating 
     connection objects c1 through c4 */
  c1=connect_groups(input,hidden);
  c2=connect_groups(hidden,output);
  c3=connect_groups(bias,hidden);
  c4=connect_groups(bias,output);

  /* add connections to our network */
  bind_connection_to_net(net,c1);
  bind_connection_to_net(net,c2);
  bind_connection_to_net(net,c3);
  bind_connection_to_net(net,c4);

  /* randomize the weights in the connection objects.
     2nd argument is weight range. */
  randomize_connections(c1,range);
  randomize_connections(c2,range);
  randomize_connections(c3,range);
  randomize_connections(c4,range);

  /* how to load and save weights */
  /*  load_weights(net,"init.weights");   */
  
  /* how to save out our weights to file 'init.weights' */
  /*  save_weights(net,"init.weights");     */

  /* load in our example set */
  examples=load_examples("xor.ex",TICKS);

  error=0.0;
  count=1;

  /* loop for ITER number of TICKSs.  This
     uses BATCH learning; we loop over all examples, then
     apply deltas */
  for(i=1;i<=ITER;i++)
    {
      /* don't apply momentum until after 10th iteration
	 (see Plaut et al. 1996) */
      if (i>10)
	{
	  for(j=0;j<net->numConnections;j++)
	    net->connections[j]->momentum=momentum;
	}

      /* loop through all examples in sequence */
      for(j=0;j<examples->numExamples;j++)
	{
	  /* get j'th example from exampleset */
	  ex=&examples->examples[j];

	  /* do forward propagation */
	  bptt_forward(net,ex);

	  /* backward pass: compute gradients */
	  bptt_compute_gradients(net,ex);

	  /* sum up error for this example */
	  error+=compute_error(net,ex);
	}

      /* quit training if our error is low enough */
      if (error < 0.01)
	break;

      /* batch learning: apply the deltas accumulated
	 from previous calls to compute_gradients1 */
      bptt_apply_deltas_momentum(net);


      /* is it TICKS to write status? */
      if (count==REP)
	{
	  /* average error over last 'count' iterations */
	  error = error/(float)count;
	  count=1;

	  /* print a message about average error so far */
	  printf("%d\t%f\n",i,error);
	}
      else count++;

      
      error=0.0;

    }

  /* done training.  time to write out results for each example */
  for(i=0;i<examples->numExamples;i++)
    {
      ex=&examples->examples[i];
      bptt_forward(net,ex);
      printf("example %d\tinputs %f\t%f\toutput %f\ttarget %f\n",
	     i,
	     get_value(ex->inputs,input->index,TICKS-1,0),
	     get_value(ex->inputs,input->index,TICKS-1,1),
	     output->outputs[TICKS-1][0],
	     get_value(ex->targets,output->index,TICKS-1,0));
    }

  /* note: the 'inputs' field above is a 3d array.  the
     first dimension is the group.  the second is the
     tick number.  the last is the unit number. */

  return 0;
}

