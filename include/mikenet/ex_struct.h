typedef struct {
  Real *value;   /* indexed by unit */
} ExpandedExample;

typedef struct {
  Real onvalue;
  Real offvalue;
  int numIndices;
  int *indices;  /* items which are on */
} SparseExample;

typedef struct {
  ExampleType type;
  ExampleClampType clampType;
  union {
    ExpandedExample expanded;
    SparseExample sparse;
  } values;
} ExampleData;

typedef struct tagExample {
  float prob;
  char *name;
  int index;
  int time;
  ExampleData ***inputs;  /* pointer, indexed by group and time */
  ExampleData ***targets; /* pointer, indexed by group and time */
  void *userData;
} Example;

