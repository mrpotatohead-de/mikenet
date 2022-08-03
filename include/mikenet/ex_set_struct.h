typedef struct {
  int numExamples;
  Example * examples;
  int *histogram;
  char *name;
  int currentExample;
} ExampleSet;
