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


#define TIME 100
#define REP 100
#define ITER 50000

int main(int argc,char *argv[])
{
  Net *net;
  Group *input,*output;
  ExampleSet *examples;
  Example *ex;
  Connections *c1;
  int i,count,j;
  Real error,tao=1.0;
  Real epsilon,range,tolerance;

  /* don't buffer output */
  setbuf(stdout,NULL);

  /* how low must error get before we quit? */
  tolerance=0.01;

  /* set random number seed to process id
     (only unix machines have getpid call) */
  mikenet_set_seed(666);

  /* a default learning rate */
  epsilon=0.5;

  
  /* default weight range */
  range=0.5;

  default_errorRadius=0.0;

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
      else if (strcmp(argv[i],"-tao")==0)
	{
	  tao=atof(argv[i+1]);
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
  
  default_tao=tao;

  if ((sizeof(Real)==4 && tolerance < 0.001) ||
      (sizeof(Real)==8 && tolerance < 0.00001))
    {
      fprintf(stderr,"careful; your tolerance is probably ");
      fprintf(stderr,"too tight for this machines precision\n");
    }
  
  /* build a network, with TIME number of time ticks */
  net=create_net(TIME);
  net->tai=1;

  net->integrationConstant=2.0/TIME;
  
  /* learning rate */
  default_epsilon=epsilon;

  /* create our groups.  
     format is: name, num of units, ticks */
  input=init_group("Input",1,TIME);
  output=init_group("Output",3,TIME);

  /* now add our groups to the network object */
  bind_group_to_net(net,input);
  bind_group_to_net(net,output);

  /* now connect our groups, instantiating */
  /* connection objects c1 through c4 */
  c1=connect_groups(input,output);

  /* add connections to our network */
  bind_connection_to_net(net,c1);

  /* randomize the weights in the connection objects.
     2nd argument is weight range. */
  randomize_connections(c1,range);

  output->taoEpsilon=0.1;

  for(i=0;i<3;i++)
    {
      c1->weights[i][0]=3.0;
      c1->frozen[i][0]=1;
    }

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
      /* get j'th example from exampleset */
      ex=&examples->examples[0];
      /* do forward propagation */
      crbp_forward(net,ex);
      
      /* backward pass: compute gradients */
      crbp_compute_gradients(net,ex);
      
      /* sum up error for this example */
      error+=compute_error(net,ex);

      crbp_update_taos(net);
      crbp_apply_deltas(net);
      
      /* is it time to write status? */
      if (count==REP)
	{
	  /* average error over last 'count' iterations */
	  error = error/(float)count;
	  count=0;
	  
	  /* print a message about average error so far */
	  printf("%d\t%f\t",i,error);
	  for(j=0;j<3;j++)
	    printf("%f ",output->taos[j]);
	  printf("\n");
	  
	  /* zero error; start counting again */
	  error=0.0;
	}
      count++;
    }
  
  printf("now evaluation\n");
  ex=&examples->examples[0];
  crbp_forward(net,ex);
  for(i=0;i<TIME;i++)
    {
      printf("time %d t1 %.3f o1 %.3f t2 %.3f o2 %.3f t3 %.3f o3 %.3f\n",
	     i,
	     1.0,
	     output->outputs[i][0],
	     2.0,
	     output->outputs[i][1],
	     0.5,
	     output->outputs[i][2]);
      
    }

  return 0;
}
