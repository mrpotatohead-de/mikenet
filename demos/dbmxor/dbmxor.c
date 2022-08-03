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
#define REP 25
#define ITER 5000

int main(int argc,char *argv[])
{
  int j;
  float e;
  int dbd=0;
  Net *net;
  float momentum=0.0;
  Group *input,*hidden,*output,*bias;
  ExampleSet *examples;
  Example *ex;
  Connections *c1,*c2,*c3,*c4;
  int i,count;
  Real error;
  int seed=666;
  Real epsilon,range,tolerance;

  /* don't buffer output */
  setbuf(stdout,NULL);
  announce_version();
  seed=getpid();

  /* how low must error get before we quit? */
  tolerance=0.01;

  /* set random number seed to process id
     (only unix machines have getpid call) */

  default_temperature=1.0;
  default_temporg=50;
  default_tempmult=0.9;
  /* a default learning rate */
  epsilon=0.1;
  
  /* default weight range */
  range=0.5;

  default_errorRadius=0.1;

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
      else if (strcmp(argv[i],"-dbd")==0)
	{
	  dbd=1;
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
      else
	{
	  fprintf(stderr,"unknown option: %s\n",argv[i]);
	  exit(-1);
	}
    }

  mikenet_set_seed(seed);
  default_momentum=0.0;
  
  if ((sizeof(Real)==4 && tolerance < 0.001) ||
      (sizeof(Real)==8 && tolerance < 0.00001))
    {
      fprintf(stderr,"careful; your tolerance is probably ");
      fprintf(stderr,"too tight for this machines precision\n");
    }
  
  /* build a network, with TIME number of time ticks */
  net=create_net(TIME);
  net->integrationConstant=0.5;
  
  /* learning rate */
  default_epsilon=epsilon;

  /* create our groups.  
     format is: name, num of units, ticks */
  input=init_group("Input",2,TIME);
  hidden=init_group("hidden",10,TIME);
  output=init_group("Output",21,TIME);

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

  /* load in our example set */
  examples=load_examples("xor.ex",TIME);

  error=0.0;
  count=0;
  /* loop for ITER number of times */
  for(i=0;i<ITER;i++)
    {

      for(j=0;j<examples->numExamples;j++)
	{
	  ex=&examples->examples[j];
	  dbm_positive(net,ex);
	  dbm_negative(net,ex);
	  dbm_update(net);
	}

      e = output->outputs[TIME-1][0] - 
	get_value(ex->targets,output->index,0,0);
      error += e * e;

      dbm_apply_deltas(net);

      
      if (count==REP)
	{
	  /* average error over last 'count' iterations */
	  error = error/(float)count;
	  count=0;

	  /* print a message about average error so far */
	  printf("%d\t%f\n",i,error);
	  /* are we done? */
	  if (error < tolerance)
	    {
	      printf("quitting... error low enough\n");
	      /* pop out of loop */
	      break;
	    }
	  /* zero error; start counting again */
	  error=0.0;
	}
      count++;
    }


  system("rm out.weights*");
  save_weights(net,"out.weights");
  return 0;
}
