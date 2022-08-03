#include <stdio.h>
#include <math.h>
#include <mikenet/simulator.h>

#include "model.h"
Net *reading;
Group *input,*hidden,*output,*phohid;
Connections *c1,*c2,*c3,*c4,*c5;
ExampleSet *reading_examples;

int build_model(void)
{
  float range;
  int i;
  /* build a network, with TIME number of time ticks */

  default_tai=1;
  reading=create_net(TIME);

  reading->integrationConstant=0.25;

  /* learning rate */
  default_epsilon=0.001;
  default_activationType=TANH_ACTIVATION;

  /* error radius */
  default_errorRadius=0.1;

  /* create our groups.  
     format is: name, num of units,  ticks */
  input=init_group("Ortho",208,TIME);
  hidden=init_group("Hidden",100,TIME);
  output=init_group("Phono",55,TIME);
  phohid=init_group("PhoHid",20,TIME);


  /* now add our groups to the network object */
  bind_group_to_net(reading,input);
  bind_group_to_net(reading,hidden);
  bind_group_to_net(reading,output);
  bind_group_to_net(reading,phohid);

  /* now connect our groups, instantiating
     connection objects c1 through c4 */
  c1=connect_groups(input,hidden);
  c2=connect_groups(hidden,output);
  c3=connect_groups(output,output);
  c4=connect_groups(output,phohid);
  c5=connect_groups(phohid,output);

  /* add connections to our network */
  bind_connection_to_net(reading,c1);
  bind_connection_to_net(reading,c2);
  bind_connection_to_net(reading,c3);
  bind_connection_to_net(reading,c4);
  bind_connection_to_net(reading,c5);

  /* randomize the weights in the connection objects.
     2nd argument is weight range. */
  range=0.1;
  randomize_connections(c1,range);
  randomize_connections(c2,range);
  randomize_connections(c3,range);
  randomize_connections(c4,range);
  randomize_connections(c5,range);

  c3->epsilon=0.001;
  c4->epsilon=0.001;
  c5->epsilon=0.001;

  precompute_topology(reading,input);
  for(i=0;i<reading->numGroups;i++)
    printf("%s %d\n",reading->groups[i]->name,
	   reading->groups[i]->whenDataLive);
  


  for(i=0;i<c3->to->numUnits;i++)
    {
      c3->weights[i][i]=0.75;
      c3->frozen[i][i]=1;
    }
  return 0;
}
