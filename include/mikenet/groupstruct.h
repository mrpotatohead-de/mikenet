typedef struct tagGroup
{
  char *name;
  int numUnits;
  int index;         /* group specifier */
  char **unitNames;
  Real ** inputs;    /* indexed by time and unit */
  Real ** outputs;   /* indexed by time and unit */
  Real ** goalOutputs;   /* indexed by time and unit, used in crbp only */
  int activationType;  /* LOGISTIC_ACTIVATION or TANH_ACTIVATION */
  int bias;  /* is it a bias */
  Real biasValue;
  int numIncoming,numOutgoing;
  Real primeOffset;
  int scaling;  /* do we scale error (SCALE_NONE or SCALE_PROB) */
  Real *z,*backz; /* used by crbp */
  Real *taos;  /* time constants */
  Real **dedx;  /* indexed over time and units */
  Real **dydtao;  /* indexed over time and units */
  Real **dxdtao;  /* indexed over time and units */
  Real *tempVector;
  Real *dedtao; 
  Real taoDecay; /* how much tao values are decayed by */
  Real taoEpsilon; /* learning rate for time constants */
  Real taoMaxMultiplier; /* what can learnable taos maximally go to? 
			    default: -1 means 1/net->integrationConstant */
  Real taoMinMultiplier; /* what can learnable taos minimally go to? */
  Real temperature;
  Real targetNoise;  /* sd for gaussian noise added to target */
  Real clampNoise;  /* sd for gaussian noise added to input clamp */
  Real activationNoise; /* sd for gaussian noise on output */
  Real inputNoise; /* sd for gaussian noise on input */
  int resetActivation;  /* reset outputs to 0 at t=0? */
  Real errorRadius;
  int errorComputation;  /* SUM_SQUARED_ERROR or CROSS_ENTROPY */
  Real **goalInputs;  /* indexed by time, unit */
  Real *storedStates;
  Real temporg,tempmult;  /* in dbm, 
			     temp = 1/(temp + temporg * (tempmult ^ t)) */
  int *delays;  /* delays for each unit: default is 1 */
  Real *exampleData; /* temp holder for example targets or clamps */
  int clampType; /* temp holder for example type */
  int errorRamp;
  Real softClampThresh; /* soft clamp will drive unit to within this value
			   of output MAX or MIN */
  Real *errorScaling;
  Real errorScale;
  int elmanContext;  /*  is this group an Elman context group? */
  Real *elmanValues;  /* values copied from the hidden unit.. used as clamp */
  struct tagGroup *elmanCopyFrom; /* which group do we copy from */
  int elmanUpdate; /* how often do we copy hidden to context */
  int whenDataLive; /* at what tick does this group have live data? */
  int (*preUpdateUnitMethod)(void *group,void *example,int t);
  int (*postUpdateUnitMethod)(void *group,void *example,int t);
  int (*preTargetSetMethod)(void *group,void *example,int t);
  int (*postTargetSetMethod)(void *group,void *example,int t);
  int (*preComputeDeDxMethod)(void *group,void *example,int t);
  int (*postComputeDeDxMethod)(void *group,void *example,int t);
  int (*postApplyExampleClampsMethod)(void *group,void *example,int t);
  void *userData;
} Group;



