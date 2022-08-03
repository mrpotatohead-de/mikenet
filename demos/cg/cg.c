#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#include <mikenet/simulator.h>

#define TIME 7
#define REP 500
int SAVE=5;
#define ITER 20000000
#define TESTS 3

int verbose=0;


float g(Net *net,ExampleSet *examples)
{
  int j;
  float error=0;
  Example *ex;
  
  for(j=0;j<examples->numExamples;j++)
    {
      ex=&examples->examples[j];
      bptt_forward(net,ex);
      bptt_compute_gradients(net,ex);
      error+=compute_error(net,ex);
    }
  return error;
}


float abscissa(float *x,float *y)
{
  float n;
  double a,b,c;
  double fa,fb,fc;
  a=x[0];
  b=x[1];
  c=x[2];
  fa=y[0];
  fb=y[1];
  fc=y[2];

  n = b - 0.5 * (((b - a) * (b - a) * (fb - fc) - (b - c)*(b - c)*(fb - fa))/
		 ((b - a) * (fb - fc) - (b - c)*(fb - fa)));
  return n;
}


float f(Net *net,ExampleSet *examples)
{
  int j;
  float error=0;
  Example *ex;
  
  for(j=0;j<examples->numExamples;j++)
    {
      ex=&examples->examples[j];
      bptt_forward(net,ex);
      error+=compute_error(net,ex);
    }
  return error;
}

#define N 7
double epsilons[N]= {0.001,0.0005,0.0002,0.0001,0.00005,0.00001,0.000005};

void
sample(Net *net,ExampleSet *examples,double *epsilons,float *e)
{
  int j,i;
  store_all_weights(net);
  for(j=0;j<N;j++)
    {
      test_step(net,epsilons[j]);
      e[j]=f(net,examples);
      if (verbose)
	printf("%g\t%f\n",
	       epsilons[j],e[j]);
      restore_all_weights(net);
      if (j>0 && e[j] > e[j-1])
	{
	  for(i=j+1;i<N;i++)
	    e[i]=1000000000.0;
	  return;
	}
    }
}

int main(int argc,char *argv[])
{
  ExampleSet *reading_examples;
  Net *reading;
  int first=1;
  Group *input,*hidden,*output,*bias,*phohid;
  char fn[255];
  int iter=ITER;
  float e[N];
  int use_cg=0;
  Real range;
  Connections *c1,*c2,*c3,*c4,*c5,*c6,*c7,*c8,*c9;
  int i,count,j,save;
  float best,bestEpsilon;
  Real error;

  setbuf(stdout,NULL);
  /* set random number seed to process id */
  mikenet_set_seed(666);

  for(i=1;i<argc;i++)
    {
      if (strcmp(argv[i],"-seed")==0)
	{
	  mikenet_set_seed(atol(argv[i+1]));
	  i++;
	}
      else if (strncmp(argv[i],"-iter",5)==0)
	{
	  iter=atoi(argv[i+1]);
	  i++;
	}
      else if (strcmp(argv[i],"-cg")==0)
	{
	  use_cg=1;
	}
      else if (strncmp(argv[i],"-verbose",4)==0)
	{
	  verbose=1;
	}
    }
  

  if (use_cg)
    printf("using cg method\n");

  /* build a network, with TIME number of time ticks */
  reading=create_net(TIME);

  
  /* learning rate */
  default_epsilon=0.01;

  /* error radius */
  default_errorRadius=0.1;

  /* create our groups.  
     format is: name, num of units,  ticks */
  input=init_group("Ortho",208,TIME);
  hidden=init_group("Hidden",50,TIME);
  output=init_group("Phono",108,TIME);
  phohid=init_group("PhoHid",20,TIME);
  bias=init_bias(1.0,TIME);

  /* now add our groups to the network object */
  bind_group_to_net(reading,input);
  bind_group_to_net(reading,hidden);
  bind_group_to_net(reading,output);
  bind_group_to_net(reading,phohid);
  bind_group_to_net(reading,bias);

  /* now connect our groups, instantiating
     connection objects c1 through c4 */
  c1=connect_groups(input,hidden);
  c2=connect_groups(hidden,output);
  c3=connect_groups(output,output);
  c4=connect_groups(output,phohid);
  c5=connect_groups(phohid,output);
  c6=connect_groups(input,output);
  c7=connect_groups(bias,hidden);
  c8=connect_groups(bias,output);
  c9=connect_groups(bias,phohid);

  /* add connections to our network */
  bind_connection_to_net(reading,c1);
  bind_connection_to_net(reading,c2);
  bind_connection_to_net(reading,c3);
  bind_connection_to_net(reading,c4);
  bind_connection_to_net(reading,c5);
  bind_connection_to_net(reading,c6);
  bind_connection_to_net(reading,c7);
  bind_connection_to_net(reading,c8);
  bind_connection_to_net(reading,c9);

  /* randomize the weights in the connection objects.
     2nd argument is weight range. */
  range=0.1;
  randomize_connections(c1,range);
  randomize_connections(c2,range);
  randomize_connections(c3,range);
  randomize_connections(c4,range);
  randomize_connections(c5,range);
  randomize_connections(c6,range);
  randomize_connections(c7,range);
  randomize_connections(c8,range);
  randomize_connections(c9,range);

  /* load in our example set */
  reading_examples=load_examples("ortho.pat",TIME);

  error=0.0;
  count=0;
  save=1;
  /* loop for ITER number of times */
  bestEpsilon=0.0;
  for(i=1;i<=iter;i++)
    {
      error=g(reading,reading_examples);
      if (use_cg)
	{
	  if (first)
	    {
	      init_cg(reading);
	      first=0;
	    }
	  else cg(reading);
	}
      
      printf("iter %d \terror %f\tepsilon %g\n",i,error,bestEpsilon);

      sample(reading,reading_examples,epsilons,e);
      best = 100000000.0;
      for(j=0;j<N;j++)
	{
	  if (e[j]<best)
	    {
	      best=e[j];
	      bestEpsilon=epsilons[j];
	    }
	}
      
      test_step(reading,bestEpsilon);
      zero_gradients(reading);
      if (save==SAVE)
	{
	  sprintf(fn,"%d_weights",i);
	  save_weights(reading,fn);
	  save=1;
	  if (i >=25)
	    SAVE=25;
	  if (i>=100)
	    SAVE=100;
	}
      else save++;
    }
  return 0;
}
