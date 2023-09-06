CC := gcc
CFLAGS := -O3 -Wall -funroll-loops
CPPFLAGS = -Iinclude/mikenet

LIB := lib/$(shell uname -m)/libmikenet.a
LIB_BLAS := lib/$(shell uname -m)/mkl/libmikenet.a
LIB_OPENMP := lib/$(shell uname -m)/openmp/libmikenet.a
LIB_BLIS := lib/$(shell uname -m)/blis/libmikenet.a

OBJDIR := src
OBJS := $(addprefix $(OBJDIR)/, \
	analyze.o \
	apply.o \
	benchmark.o \
	bptt.o \
	crbp.o \
	dbm.o \
	dotprod.o \
	elman.o \
	error.o \
	example.o \
	fastexp.o \
	linesearch.o \
	matrix.o \
	net.o \
	parallel.o \
	random.o \
	stats.o \
	tools.o \
	token.o \
	weights.o \
)

.PHONY: clean help lib lib_mkl lib_openmp lib_blis

lib: $(LIB)		## build Mikenet as static library
	-@$(RM) -r $(OBJS)

$(LIB): $(OBJS)
	@mkdir -p $(@D)
	$(AR) rcs $@ $^

lib_mkl: CPPFLAGS += -I$(MKLROOT)/include -DUSE_BLAS
lib_mkl: $(LIB_BLAS)	## build Mikenet as static library with MKL optimization
	-@$(RM) -r $(OBJS)

$(LIB_BLAS): $(OBJS)
	@mkdir -p $(@D)
	$(AR) rcs $@ $^

lib_openmp: CPPFLAGS += -fopenmp -DUSE_OPENMP
lib_openmp: $(LIB_OPENMP)	## build Mikenet as static library with openMP
	-@$(RM) -r $(OBJS)

$(LIB_OPENMP): $(OBJS)
	@mkdir -p $(@D)
	$(AR) rcs $@ $^

lib_blis: CPPFLAGS += -fopenmp -DUSE_AMDBLIS -I/home/nm6114083/my_ws/blis/include/zen3
lib_blis: $(LIB_BLIS)	## build Mikenet as static library with AMD BLIS
	-@$(RM) -r $(OBJS)

$(LIB_BLIS): $(OBJS)
	@mkdir -p $(@D)
	$(AR) rcs $@ $^

clean:	## delete Mikenet libraries and the object files
	-@$(RM) -rv $(OBJS) $(LIB) $(LIB_BLAS) $(LIB_OPENMP) $(LIB_BLIS)

help:   ## print this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36mmake %-20s\033[0m %s\n", $$1, $$2}'

# implicit rule
# %.o: %.c
# 	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

# DO NOT DELETE

analyze.o: const.h net.h groupstruct.h connstruct.h netstruct.h example.h
analyze.o: ex_struct.h ex_set_struct.h tools.h random.h analyze.h
apply.o: const.h net.h groupstruct.h connstruct.h netstruct.h weights.h
apply.o: tools.h
benchmark.o: const.h net.h groupstruct.h connstruct.h netstruct.h example.h
benchmark.o: ex_struct.h ex_set_struct.h tools.h random.h benchmark.h bptt.h
benchmark.o: crbp.h
bptt.o: const.h net.h groupstruct.h connstruct.h netstruct.h example.h
bptt.o: ex_struct.h ex_set_struct.h weights.h bptt.h random.h tools.h error.h
crbp.o: const.h net.h groupstruct.h connstruct.h netstruct.h example.h
crbp.o: ex_struct.h ex_set_struct.h weights.h bptt.h crbp.h random.h tools.h
crbp.o: error.h dotprod.h matrix.h
dbm.o: const.h net.h groupstruct.h connstruct.h netstruct.h example.h
dbm.o: ex_struct.h ex_set_struct.h weights.h dbm.h bptt.h random.h tools.h
dbm.o: error.h
dotprod.o: const.h dotprod.h
elman.o: const.h net.h groupstruct.h connstruct.h netstruct.h example.h
elman.o: ex_struct.h ex_set_struct.h weights.h bptt.h random.h tools.h
elman.o: elman.h
error.o: const.h net.h groupstruct.h connstruct.h netstruct.h example.h
error.o: ex_struct.h ex_set_struct.h error.h tools.h
example.o: const.h net.h groupstruct.h connstruct.h netstruct.h random.h
example.o: tools.h example.h ex_struct.h ex_set_struct.h
fastexp.o: fastexp.h
linesearch.o: const.h net.h groupstruct.h connstruct.h netstruct.h tools.h
linesearch.o: weights.h linesearch.h
matrix.o: const.h matrix.h
net.o: const.h net.h groupstruct.h connstruct.h netstruct.h tools.h random.h
parallel.o: const.h net.h groupstruct.h connstruct.h netstruct.h example.h
parallel.o: ex_struct.h ex_set_struct.h tools.h weights.h error.h crbp.h
parallel.o: parallel.h
random.o: random.h
stats.o: stats.h
token.o: token.h
tools.o: const.h net.h groupstruct.h connstruct.h netstruct.h example.h
tools.o: ex_struct.h ex_set_struct.h tools.h random.h fastexp.h
weights.o: const.h net.h groupstruct.h connstruct.h netstruct.h tools.h
weights.o: weights.h random.h
dotprod.o: const.h
example.o: ex_struct.h ex_set_struct.h
net.o: groupstruct.h connstruct.h netstruct.h
