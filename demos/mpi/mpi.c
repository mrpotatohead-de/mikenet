
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef unix
#include <unistd.h>
#endif

#include <mikenet/simulator.h>
#include "simconfig.h"
#include "model.h"
#include "hearing.h"
#include "reading.h"
#include "train.h"

ExampleSet *hearing_examples,*speaking_examples;
ExampleSet *sem_examples,*phono_examples;
int myid;


/* what is the error if we move the net in the direction of lr */
/* assumes weights have been stored already, and gradient is computed  */
float sample(float lr)
{
  float e1,e2,e3,e4;
  /* move in the direction */
  if (myid==0)
    test_step(hearing,lr);
  parallel_broadcast_weights(hearing);
  e1=parallel_f(ps,hearing_examples,0.8);
  e2=parallel_f(sp,speaking_examples,0.8);
  e3=parallel_f(sem,sem_examples,0.2);
  e4=parallel_f(phono,phono_examples,0.2);
  return e1+e2+e3+e4;
}

int main(int argc,char *argv[])
{
  char cmd[255];
  int hcount=0,scount=0;
  float herr=0,serr=0,dice;
  Example *ex;
  int do_cg=0;
  float epsilon;
  int seed=1,first=1;
  float e,e0,lr,lrPrev;
  int start=1,i,j;
  char  loadFile[255],fn[255];
  setbuf(stdout,NULL);

  parallel_init(&argc,&argv);

  if (myid==0)
    {
      announce_version();
      system("hostname");
    }
  printf("pid %d\n",getpid());


  loadFile[0]=0;

  /* what are the command line arguments? */
  for(i=1;i<argc;i++)
    {
      if (strcmp(argv[i],"-seed")==0)
	{
	  seed=atoi(argv[i+1]);
	  i++;
	}
      else if (strcmp(argv[i],"-start")==0)
	{
	  start=atoi(argv[i+1]);
	  i++;
	}
      else if (strncmp(argv[i],"-epsilon",5)==0)
	{
	  epsilon=atof(argv[i+1]);
	  i++;
	}
      else if (strncmp(argv[i],"-load",5)==0)
	{
	  strcpy(loadFile,argv[i+1]);
	  i++;
	}
      else
	{
	  fprintf(stderr,"unknown argument: %s\n",argv[i]);
	  exit(1);
	}
    }

  default_epsilon=0.05;


  mikenet_set_seed(seed);

  build_hearing_model(SAMPLES);

  /* load in our example set */
  phono_examples=load_examples("phono.pat",TICKS);
  sem_examples=load_examples("sem.pat",TICKS);
  hearing_examples=load_examples("ps.pat",TICKS);
  speaking_examples=load_examples("sp.pat",TICKS);

  phono_examples->numExamples=500;
  sem_examples->numExamples=500;
  hearing_examples->numExamples=500;
  speaking_examples->numExamples=500;


  myid=parallel_proc_id();


#ifdef DO_ONLINE_PRE_TRAIN
  if (start==1)
    {
      for(i=1;i<=10000;i++)
	{
	  dice = mikenet_random();
	  if (dice <=0.2)
	    {
	      ex=get_random_example(phono_examples);
	      crbp_forward(phono,ex);
	      crbp_compute_gradients(phono,ex);
	      crbp_apply_deltas(phono);
	    }
	  else if (dice <= 0.5)
	    {
	      ex=get_random_example(hearing_examples);
	      crbp_forward(ps,ex);
	      crbp_compute_gradients(ps,ex);
	      herr += compute_error(ps,ex);
	      crbp_apply_deltas(ps);
	    }
	  else if (dice <= 0.7)
	    {
	      ex=get_random_example(sem_examples);
	      crbp_forward(sem,ex);
	      crbp_compute_gradients(sem,ex);
	      crbp_apply_deltas(sem);
	    }
	  else
	    {
	      ex=get_random_example(speaking_examples);
	      crbp_forward(sp,ex);
	      crbp_compute_gradients(sp,ex);
	      serr+=compute_error(sp,ex);
	      crbp_apply_deltas(sp);
	    }

	  if (i % 100 == 0)
	    {
	      printf("%d hear %f speak %f\n",i,
		     herr/hcount,serr/scount);
	      herr=0.0;
	      serr=0.0;
	      hcount=0;
	      scount=0;
	    }
	}
      sprintf(fn,"s%d_online_weights",seed);
      save_weights(hearing,fn);
    }
#endif

  parallel_broadcast_weights(hearing);


  do_cg=1;
  if (do_cg && myid==0)
    printf("USING CG\n");

  /* loop for ITER number of times */
  for(i=1;i<5;i++)
    {
      parallel_sync();
      store_all_weights(hearing);
      e=parallel_g(ps,hearing_examples,0.8) +
	parallel_g(sp,speaking_examples,0.8) +
	parallel_g(sem,sem_examples,0.2) +
	parallel_g(phono,phono_examples,0.2);
      
      if(do_cg)
	{
	  if (first)
	    init_cg(hearing);
	  else
	    cg(hearing);
	  first=0;
	}

      e0=e;
      if (myid==0)
	printf("%d e0: %f\n",i,e);
      lr = 0.2/hearing_examples->numExamples;
      lrPrev=lr/10;
      for(j=1;j<10;j++)
	{
	  e=sample(lr);
	  if (myid==0)
	    printf("\t\t%d %f %f\n",j,e,lr);
	  if (e>e0)
	    {
	      if (myid==0)
		test_step(hearing,lrPrev);
	      parallel_broadcast_weights(hearing);
	      break;
	    }
	  e0=e;
	  lrPrev=lr;
	  lr *= 1.7;
	}
      zero_gradients(hearing);
      if (i % 5==0 && myid==0)
	{
	  sprintf(fn,"s%d_%d_weights",seed,i);
	  save_weights(hearing,fn);
	}
    }
  parallel_finish();
  return 0;
}
