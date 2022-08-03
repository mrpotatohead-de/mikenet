typedef struct
{
  int numConnections; /* how many connection sets do we have */
  Connections **connections;
  int numGroups;
  Group **groups;
  int pid; /* process id */
  int time;  /* how many ticks (max) */
  int tai; /* time average inputs?  */
  Real integrationConstant;
  int runfor;  /* how long to run for */
  int (*preForwardMethod)(void *n,void *ex,int t);
  int (*postForwardMethod)(void *n,void *ex,int t);
  int (*preComputeGradientsMethod)(void *n,void *ex,int t);
  int (*postComputeGradientsMethod)(void *n,void *ex,int t);
  int t; /* which tick are we on */
  void *userData;
} Net;
