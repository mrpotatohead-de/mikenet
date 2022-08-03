typedef struct _tagConnections
{
  Real epsilon;
  Real momentum;
  Group * from, *to;  /* groups from and to */
  int * numIncoming;
  int ** incomingUnits;
  unsigned char ** frozen;   /* indexed by toUnit and fromUnit */
  Real ** weights;  /* ditto */
  Real ** backupWeights;  /* normally null, unless storeWeights called */
  Real weightNoise;
  int weightNoiseType;  /* NO_NOISE, ADDITIVE_NOISE or MULTIPLICATIVE_NOISE */
  Real ** gradients; /* ditto */  
  Real ** prevDeltas;
  Real **h,**g;
  Real ** dbdWeight;
  Real dbdUp,dbdDown;  /* dbd up and down factor */
  int locked;  /* is it locked? */
  int (*preForwardPropagateMethod)(void *c,void *ex,int t);
  int (*postForwardPropagateMethod)(void *c,void *ex,int t);
  int errorType; /* do we use normal error, or Oja's rule */
  Real * contribution; /* dimensioned by TO group: input to that group */
  Real scaleGradients;
  void *userData;
} Connections;  
