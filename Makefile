CC= gcc
MPICC= mpicc
RM= rm -vf
CFLAGS= -Wall -g -O2
OPENMPFLAG= -fopenmp
LIBS= -lm

PROGFILES= icosahedral_laplace

main: icosahedral_laplace.c
	$(MPICC) $(CFLAGS) $(LIBS) icosahedral_laplace.c -o $(PROGFILES)

clean:
	$(RM) $(PROGFILES)
