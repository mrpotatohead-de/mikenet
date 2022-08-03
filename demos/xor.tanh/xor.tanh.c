/* to compile this, your MIKENET_DIR environment variable
   must be set to the appropriate directory.  putting this
   at the bottom of your .cshrc file will do the trick:

   setenv MIKENET_DIR ~mharm/mikenet/default/


   This demo program solves the xor problem.  

*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef unix
#include <unistd.h>
#endif

#include <mikenet/simulator.h>


#define TIME 3
#define REP 100
#define ITER 1000

int main(int argc,char *argv[])
{
  Net *net;
  Group *input,*hidden,*output,*bias;
  ExampleSet *examples;
  Example *ex;
  Connections *c1,*c2,*c3,*c4;
  int i,count,j;
  Real error;
  Real epsilon,range,tolerance;

  /* don't buffer output */
  setbuf(stdout,NULL);
  announce_version();

  /* how low must error get before we quit? */
  tolerance=0.01;

  /* set random number seed to process id
     (only unix machines have getpid call) */
#ifdef unix
  mikenet_set_seed(getpid()); 
#endif

  /* a default learning rate */
  epsilon=0.5;
  
  /* default weight range */
  range=0.5;

  default_errorRadius=0.1;

  /* what are the command line arguments? */
  for(i=1;i<argc;i++)
    {
      if (strcmp(argv[i],"-seed")==0)
	{
	  mikenet_set_seed(atol(argv[i+1]));
	  i++;
	}
      else if (strncmp(argv[i],"-epsilon",5)==0)
	{
	  epsilon=atof(argv[i+1]);
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
      else if (strncmp(argv[i],"-tolerance",4)==0)
	{
	  tolerance=atof(argv[i+1]);
	  i++;
	}
    }
  
  if ((sizeof(Real)==4 && tolerance < 0.001) ||
      (sizeof(Real)==8 && tolerance < 0.00001))
    {
      fprintf(stderr,"careful; your tolerance is probably ");
      fprintf(stderr,"too tight for this machines precision\n");
    }
  
  /* build a network, with TIME number of time ticks */

  default_activationType=TANH_ACTIVATION;
  default_errorComputation=CROSS_ENTROPY_ERROR;
  net=create_net(TIME);
  
  /* learning rate */
  default_epsilon=epsilon;

  /* create our groups.  
     format is: name, num of units, ticks */
  input=init_group("Input",2,TIME);
  hidden=init_group("hidden",2,TIME);
  output=init_group("Output",1,TIME);

  /* bias is special.  format is: value, ticks */
  bias=init_bias(1.0,TIME);   

  /* now add our groups to the network object */
  bind_group_to_net(net,input);
  bind_group_to_net(net,hidden);
  bind_group_to_net(net,output); 
  bind_group_to_net(net,bias); 

  /* now connect our groups, instantiating */
  /* connection objects c1 through c4 */
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
  
  /* erase old initial weight file */
  /*  system("rm -f init.weights.Z");      */

  /* save out our weights to file 'init.weights' */
  /*  save_weights(net,"init.weights");     */

  /* load in our example set */
  examples=load_examples("xor.ex",TIME);

  error=0.0;
  count=0;
  /* loop for ITER number of times */
  for(i=0;i<ITER;i++)
    {
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
      /* batch learning: apply the deltas accumulated
	 from previous calls to compute_gradients1 */
      bptt_apply_deltas(net);

      /* is it time to write status? */
      if (count==REP)
	{
	  /* average error over last 'count' iterations */
	  error = error/(float)count;
	  count=0;

	  /* print a message about average error so far */
	  printf("%d\t%f\n",i,error);
#ifdef FOO
	  /* are we done? */
	  if (error < tolerance)
	    {
	      printf("quitting... error low enough\n");
	      /* pop out of loop */
	      break;
	    }
#endif
	  /* zero error; start counting again */
	  error=0.0;
	}
      count++;
    }

  /* write out results for each example */
  for(i=0;i<examples->numExamples;i++)
    {
      ex=&examples->examples[i];
      bptt_forward(net,ex);
      printf("example %d\tinputs %f\t%f\toutput %f\ttarget %f\n",
	     i,
	     get_value(ex->inputs,input->index,TIME-1,0),
	     get_value(ex->inputs,input->index,TIME-1,1),
	     output->outputs[TIME-1][0],
	     get_value(ex->targets,output->index,TIME-1,0));
    }
  return 0;
}
