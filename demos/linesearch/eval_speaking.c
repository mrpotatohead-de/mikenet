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
#include "model.h"
#include "hearing.h"
#include "reading.h"
#include "simconfig.h"

#define REP 100
#define ITER 10000000

typedef struct
{
  char ch;
  Real vector[PHO_FEATURES];
} Phoneme;

Phoneme phonemes[50];
int phocount=0;

int symbol_hash[255];

void get_name(char *tag, char *name)
{
  char *p;
  p=strstr(tag,"Word:");
  p+= 5;
  p=strtok(p," \t\n");
  strcpy(name,p);
}

void load_phonemes()
{
  FILE * f;
  char line[255],*p;
  int i,x;
  f=fopen("mapping","r");
  if (f==NULL)
    {
      fprintf(stderr,"no mapping file\n");
      exit(1);
    }
  x=0;
  fgets(line,255,f);
  while(!feof(f))
    {
      p=strtok(line," \t\n");
      if (p[0]=='-')
	p[0]='_';
      phonemes[phocount].ch=p[0];
      symbol_hash[(unsigned int)(p[0])]=x++;
      for(i=0;i<PHO_FEATURES;i++)
	{
	  p=strtok(NULL," \t\n");
	  if (strcmp(p,"NaN")==0)
	    phonemes[phocount].vector[i]= -10;
	  else 
	    phonemes[phocount].vector[i]= atof(p);
	}
      phocount++;
      fgets(line,255,f);
    }
  fclose(f);
}

float euclid_distance(Real *x1,Real *x2)
{
  float d=0,r;
  int i;
  for(i=0;i<PHO_FEATURES;i++)
    {
      r = x1[i] - x2[i];
      d += r * r;
    }
  return d;
}
      
void euclid(Real *v,char *out)
{
  int i,j;
  int nearest_item;
  float error=0;
  float nearest_distance=1000000000.0,d;

  for(i=0;i<PHO_SLOTS;i++)
    {
      nearest_item=-1;
      for(j=0;j<phocount;j++)
	{
	  d=euclid_distance(&v[i*PHO_FEATURES],phonemes[j].vector);
	  if ((nearest_item == -1) ||
	      (d < nearest_distance))
	    {
	      nearest_item=j;
	      nearest_distance=d;
	    }
	}
      error += d;
      out[i]=phonemes[nearest_item].ch;
    }
  out[PHO_SLOTS]=0;
}


int main(int argc,char *argv[])
{
  char euclid_output[100],target_output[100];
  Real sse;
  FILE *f;
  Real clampNoise=0.1;
  char feature[1000][30];
  char name[255];
  int time=4;
  int wrongs=0,wrong=0,wrongtot;
  char weightFile[4000];
  int reset=1;
  ExampleSet *examples;
  Example *ex;
  int i,count;
  Real error;
  Real epsilon,range;
  int runfor=TICKS;


  f=fopen("key","r");
  if (f==NULL)
    {
      fprintf(stderr,"can't open key file\n");
      exit(-1);
    }

  i=0;
  fscanf(f,"%s",feature[i++]);
  while(!feof(f))
    fscanf(f,"%s",feature[i++]);
  fclose(f);

  /* don't buffer output */
  setbuf(stdout,NULL);

  /* set random number seed to process id
     (only unix machines have getpid call) */
#ifdef unix
  mikenet_set_seed(getpid()); 
#endif

  /* a default learning rate */
  epsilon=0.01;
  
  /* default weight range */
  range=0.01;

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
      else if (strcmp(argv[i],"-clampNoise")==0)
	{
	  clampNoise=atof(argv[i+1]);
	  i++;
	}
      else if (strncmp(argv[i],"-noreset",5)==0)
	{
	  reset=0;
	}
      else if (strncmp(argv[i],"-runfor",5)==0)
	{
	  runfor=atoi(argv[i+1]);
	  i++;
	}
      else if (strncmp(argv[i],"-time",5)==0)
	{
	  time=atoi(argv[i+1]);
	  i++;
	}
      else if (strcmp(argv[i],"-range")==0)
	{
	  range=atof(argv[i+1]);
	  i++;
	}
      else if ((strncmp(argv[i],"-weight",5)==0) ||
	       (strncmp(argv[i],"-load",5)==0))
	{
	  strcpy(weightFile,argv[i+1]);
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

  build_hearing_model();
  hearing->integrationConstant=(float)time/(float)TICKS;
  hearing->runfor=runfor;
  sp->integrationConstant=(float)time/(float)TICKS;
  sp->runfor=runfor;

  examples=load_examples("sp.pat",TICKS);
  load_weights(hearing,weightFile);

  load_phonemes();

  error=0.0;
  count=1;
  wrongs=0;
  wrongtot=0;
  /* loop for ITER number of times */
  for(i=0;i<examples->numExamples;i++)
    {
      /* get j'th example from exampleset */
      ex=&examples->examples[i];
      get_name(ex->name,name);
      printf("%s\t",name);
      /* do forward propagation */
      crbp_forward(sp,ex);
      wrong=0;
      sse=0.0;
      euclid(phonology->outputs[runfor-1],euclid_output);
      printf("%s\t",euclid_output);
      euclid(ex->targets[phonology->index][runfor-1],target_output);
      printf("%s\t",target_output);
      if (strcmp(target_output,euclid_output)!=0)
	{
	  printf("[WRONG]\n");
	  wrongs++;
	}
      else printf("\n");
    }
  printf("%d wrong\n",wrongs);
  return 0;
}
