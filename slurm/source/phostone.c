// Normal compile
// Intel:
// mpiicc -fopenmp phostone.c -o phostone
// gcc:
// mpicc  -fopenmp phostone.c -o phostone
//
// To compile without openmp
// Intel:
// mpiicc -fopenmp-stubs phostone.c -o purempi
// gcc:
// mpicc  -DSTUBS        phostone.c -o purempi
//
//
#include <unistd.h>
#include <string.h>
#include <omp.h>

#ifdef NOMPI
#define MPI_MAX_LIBRARY_VERSION_STRING 32
#define MPI_MAX_PROCESSOR_NAME 32 
#define MPI_Comm int
#define MPI_COMM_WORLD 0
#define MPI_INT 0
#define MPI_CHAR 0
#define MPI_DOUBLE 0
#define MPI_Status int
void MPI_Get_library_version(char *version, int *vlan) {strcpy(version,"NONE");*vlan=4;};
void MPI_Comm_size(int c, int *numprocs)  {*numprocs=1;};
void MPI_Comm_rank(int c, int *numprocs)  {*numprocs=0;};
void MPI_Get_processor_name(char *lname, int *resultlen) { gethostname(lname,MPI_MAX_PROCESSOR_NAME); }
void MPI_Barrier(int c){}
void MPI_Finalize(){}
void MPI_Send(void *s ,int c, int t, int f, int tg, int com ){}
void MPI_Recv(void *s ,int c, int t, int f, int tg, int com ,void *stat){}
void MPI_Bcast(void *r, int c, int t, int f, int com){}
void MPI_Comm_split(int oc, int mycolor, int myid, int *node_comm){*node_comm=0;}
void MPI_Init(int *argc, char ***argv){}
double MPI_Wtime() { return omp_get_wtime();}
#else
#include <mpi.h>
#endif

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <time.h>
#include <utmpx.h>

// which processor on a node will
// print env if requested
#ifndef PID
#define PID 0
#endif

void dothreads(int full, char *myname, int myid, int mycolor, int new_id);

char *trim(char *s);
void slowit(int nints, int val);
int node_color();
int sched_getcpu();

void ptime()
{
  time_t rawtime;
  struct tm *timeinfo;
  char buffer[80];

  time(&rawtime);
  timeinfo = localtime(&rawtime);
  strftime(buffer, 80, "%c", timeinfo);
  // puts (buffer);
  printf("%s\n", buffer);
}
int findcore()
{
  int cpu;

#ifdef __APPLE__
  cpu = -1;
#else
  cpu = sched_getcpu();
#endif
  return cpu;
}

int str_upr(char *cstr)
{
  char *str = cstr;

  for (; *str; str++)
    {
      if (isalpha(*str))
        if (*str >= 'a')
          {
            *str += 'A' - 'a';
          }
    }
  return 0;
}

int str_low(char *cstr)
{
  char *str = cstr;

  for (; *str; str++)
    {
      if (isalpha(*str))
        if (*str < 'a')
          {
            *str += 'a' - 'A';
          }
    }
  return 0;
}

void dohelp();
void dohelp()
{
  /************************************************************
   * This is a glorified hello world program. Each processor
   * prints name, rank, and other information as described below.
   * ************************************************************/
  printf("phostname arguments:\n");
  printf("          -h : Print this help message\n");
  printf("\n");
  printf("no arguments : Print a list of the nodes on which the command is "
         "run.\n");
  printf("\n");
  printf(" -f or -1    : Same as no argument but print MPI task id and Thread "
         "id\n");
  printf("               If run with OpenMP threading enabled OMP_NUM_THREADS "
         "> 1\n");
  printf("               there will be a line per MPI task and Thread.\n");
  printf("\n");
  printf(" -F or -2    : Add columns to tell first MPI task on a node and and "
         "the\n");
  printf("               numbering of tasks on a node. (Hint: pipe this output "
         "in\n");
  printf("               to sort -r\n");
  printf("\n");
  printf(" -E or -B    : Print thread info at 'E'nd of the run or 'B'oth the "
         "start and end\n");
  printf("\n");
  printf(" -a          : Print a listing of the environmental variables passed "
         "to\n");
  printf("               MPI task. (Hint: use the -l option with SLURM to "
         "prepend MPI\n");
  printf("               task #.)\n");
  printf("\n");
  printf(" -s ######## : Where ######## is an integer.  Sum a bunch on "
         "integers to slow\n");
  printf("               down the program.  Should run faster with multiple "
         "threads.\n");
  printf("\n");
  printf(" -t ######## : Where is a time in seconds.  Sum a bunch on integers "
         "to slow\n");
  printf("               down the program and run for at least the given "
         "seconds.\n");
  printf("\n");
  printf(" -T          : Print time/date at the beginning/end of the run.\n");
  printf("\n");
}
/* valid is used to get around an issue in some versions of
 * MPI that screw up the environmnet passed to programs. Its
 * usage is not recommended.  See:
 * https://wiki.sei.cmu.edu/confluence/display/c/MEM10-C.+Define+and+use+a+pointer+validation+function
 *
 * "The valid() function does not guarantee validity; it only
 * identifies null pointers and pointers to functions as invalid.
 * However, it can be used to catch a substantial number of
 * problems that might otherwise go undetected."
 */
int valid(void *p)
{
  extern char _etext;

  return (p != NULL) && ((char *)p > &_etext);
}
char f1234[128], f1235[128], f1236[128];

int main(int argc, char **argv, char *envp[])
{
  char *eql;
  int myid, numprocs, resultlen;
  int mycolor, new_id, new_nodes;
  int i, k;
  MPI_Comm node_comm;
  char lname[MPI_MAX_PROCESSOR_NAME];
  //#ifdef MPI_MAX_LIBRARY_VERSION_STRING
  char version[MPI_MAX_LIBRARY_VERSION_STRING];
  //#else
  //    char version[40];
  //#endif
  char *myname, *cutit;
  int full, envs, iarg, tn, nt, help, slow, vlan, wait, dotime, when;
  int nints;
  double t1, t2, dt;

  /* Format statements */
  //    char *f1234="%4.4d      %4.4d    %18s        %4.4d         %4.4d
  //    %4.4d\n"; char *f1235="%s %4.4d %4.4d\n"; char *f1236="%s\n";
  strcpy(f1234, "%4.4d      %4.4d    %18s        %4.4d         %4.4d  %4.4d\n");
  strcpy(f1235, "%s %4.4d %4.4d\n");
  strcpy(f1236, "%s\n");
  MPI_Init(&argc, &argv);
  //#ifdef MPI_MAX_LIBRARY_VERSION_STRING
  MPI_Get_library_version(version, &vlan);
  //#else
  //    sprintf(version,"%s","UNDEFINED - consider upgrading");
  //#endif
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Get_processor_name(lname, &resultlen);
  /* Get rid of "stuff" from the processor name. */
  myname = trim(lname);
  /* The next line is required for BGQ because the MPI task ID
     is encoded in the processor name and we don't want it. */
  if (strrchr(myname, 32))
    myname = strrchr(myname, 32);
  /* Here we cut off the tail of node name, Summit in this case */
  cutit = strstr(myname, ".rc.int.colorado.edu");
  if (cutit)
    cutit[0] = (char)0;
  slow = 0;
  wait = 0;
  /* read in command line args from task 0 */
  if (myid == 0)
    {
      full = 0;
      envs = 0;
      help = 0;
      dotime = 0;
      when = 1;
      if (argc > 1)
        {
          for (iarg = 1; iarg < argc; iarg++)
            {
              if ((strcmp(argv[iarg], "-h") == 0) ||
                  (strcmp(argv[iarg], "--h") == 0) ||
                  (strcmp(argv[iarg], "-help") == 0))
                help = 1;
              /**/
              if ((strcmp(argv[iarg], "-f") == 0) || (strcmp(argv[iarg], "-1") == 0))
                full = 1;
              /**/
              if ((strcmp(argv[iarg], "-F") == 0) || (strcmp(argv[iarg], "-2") == 0))
                full = 2;
              /**/
              if (strcmp(argv[iarg], "-s") == 0)
                slow = 1;
              /**/
              if (strcmp(argv[iarg], "-t") == 0)
                wait = 1;
              /**/
              if (strcmp(argv[iarg], "-a") == 0)
                envs = 1;
              /**/
              if (strcmp(argv[iarg], "-T") == 0)
                dotime = 1;

              if (strcmp(argv[iarg], "-B") == 0)
                when = 3;
              if (strcmp(argv[iarg], "-E") == 0)
                when = 2;
            }
        }
    }
  /* send info to all tasks, if doing help doit and quit */
  MPI_Bcast(&help, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (help == 1)
    {
      if (myid == 0)
        dohelp();
      MPI_Finalize();
      exit(0);
    }
  MPI_Bcast(&full, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&envs, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&when, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (myid == 0 && dotime == 1)
    ptime();
  if (myid == 0 && full == 2)
    {
      printf("MPI VERSION %s\n", version);
      printf("task    thread             node name  first task    # on node  "
             "core\n");
    }
  /*********/
  /* The routine NODE_COLOR will return the same value for all mpi
     tasks that are running on the same node.  We use this to create
     a new communicator from which we get the numbering of tasks on
     a node. */
  //    NODE_COLOR(&mycolor);
  mycolor = node_color();
  MPI_Comm_split(MPI_COMM_WORLD, mycolor, myid, &node_comm);
  MPI_Comm_rank(node_comm, &new_id);
  MPI_Comm_size(node_comm, &new_nodes);
  tn = -1;
  nt = -1;
  /* Here we print out the information with the format and
     verbosity determined by the value of full. We do this
     a task at a time to "hopefully" get a bit better formatting. */
  for (i = 0; i < numprocs; i++)
    {
      MPI_Barrier(MPI_COMM_WORLD);
      if (i != myid)
        continue;
      if (when == 3)
        str_low(myname);
      if (when != 2)
        dothreads(full, myname, myid, mycolor, new_id);

      /* here we print out the environment in which a MPI task is running */
      /* We try to determine if the passed environment is valid but sometimes
       * it just does not work and this can crash.  Try taking out myid==0
       * and setting PID to a nonzero value.
       */
      // if (envs == 1 && new_id==1) {
      if (envs == 1 && (myid == PID || myid == 0))
        {
          k = 0;
          if (valid(envp) == 1)
            {
              // while(envp[k]) {
              while (valid(envp[k]) == 1)
                {
                  if (strlen(envp[k]) > 3)
                    {
                      eql = strchr(envp[k], '=');
                      if (eql == NULL)
                        break;
                      printf("? %d %s\n", myid, envp[k]);
                    }
                  else
                    {
                      break;
                    }
                  // printf("? %d %d\n",myid,k);
                  k++;
                }
            }
          else
            {
              printf("? %d %s\n", myid, "Environmnet not set");
            }
        }
    }
  if (myid == 0)
    {
      dt = 0;
      if (wait)
        {
          slow = 0;
          for (iarg = 1; iarg < argc; iarg++)
            {
              // printf("%s\n",argv[iarg]);
              if (atof(argv[iarg]) > 0)
                dt = atof(argv[iarg]);
            }
        }
    }
  MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (dt > 0)
    {
      nints = 100000;
      t1 = MPI_Wtime();
      t2 = t1;
      while (dt > t2 - t1)
        {
          for (i = 1; i <= 1000; i++)
            {
              slowit(nints, i);
            }
          t2 = MPI_Wtime();
        }
      if (myid == 0)
        printf("total time %10.3f\n", t2 - t1);
      nints = 0;
    }
  if (myid == 0)
    {
      nints = 0;
      if (slow == 1)
        {
          for (iarg = 1; iarg < argc; iarg++)
            {
              if (atol(argv[iarg]) > 0)
                nints = atoi(argv[iarg]);
            }
        }
    }
  MPI_Bcast(&nints, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (nints > 0)
    {
      t1 = MPI_Wtime();
      for (i = 1; i <= 1000; i++)
        {
          slowit(nints, i);
        }
      t2 = MPI_Wtime();
      if (myid == 0)
        printf("total time %10.3f\n", t2 - t1);
    }

  if (myid == 0 && dotime == 1)
    ptime();
  if (when > 1)
    {
      for (i = 0; i < numprocs; i++)
        {
          MPI_Barrier(MPI_COMM_WORLD);
          if (i != myid)
            continue;
          if (when == 3)
            str_upr(myname);
          dothreads(full, myname, myid, mycolor, new_id);
        }
    }
  MPI_Finalize();
  return 0;
}

char *trim(char *s)
{
  int i = 0;
  int j = strlen(s) - 1;
  int k = 0;

  while (isspace(s[i]) && s[i] != '\0')
    i++;

  while (isspace(s[j]) && j >= 0)
    j--;

  while (i <= j)
    s[k++] = s[i++];

  s[k] = '\0';

  return s;
}

/*
   ! return a integer which is unique to all mpi
   ! tasks running on a particular node.  It is
   ! equal to the id of the first MPI task running
   ! on a node.  This can be used to create
   ! MPI communicators which only contain tasks on
   ! a node.

 */
int node_color()
{
  int mycol;
  MPI_Status status;
  int xchng, i, n2, myid, numprocs;
  int nlen;
  int ie;
  char *pch;
  char name[MPI_MAX_PROCESSOR_NAME + 1];
  char nlist[MPI_MAX_PROCESSOR_NAME + 1];

  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Get_processor_name(name, &nlen);
  pch = strrchr(name, ' ');
  if (pch)
    {
      ie = strlen(pch + 1);
      memmove(&name[0], pch + 1, ie + 1);
      memmove(&nlist[0], pch + 1, ie + 1);
    }
  else
    {
      strcpy(nlist, name);
    }
  mycol = myid;
  n2 = 1;
  while (n2 < numprocs)
    {
      n2 = n2 * 2;
    }
  for (i = 1; i <= n2 - 1; i++)
    {
      xchng = i ^ myid;
      if (xchng <= (numprocs - 1))
        {
          if (myid < xchng)
            {
              MPI_Send(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, xchng, 12345,
                       MPI_COMM_WORLD);
              MPI_Recv(nlist, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, xchng, 12345,
                       MPI_COMM_WORLD, &status);
            }
          else
            {
              MPI_Recv(nlist, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, xchng, 12345,
                       MPI_COMM_WORLD, &status);
              MPI_Send(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, xchng, 12345,
                       MPI_COMM_WORLD);
            }
          if (strcmp(nlist, name) == 0 && xchng < mycol)
            mycol = xchng;
        }
      else
        {
          /* skip this stage */
        }
    }
  return mycol;
}

void slowit(int nints, int val)
{
  int *block;
  long i, sum;

#ifdef VERBOSET
  double t2, t1;
  t1 = MPI_Wtime();
#endif
  block = (int *)malloc(nints * sizeof(int));
#pragma omp parallel for
  for (i = 0; i < nints; i++)
    {
      block[i] = val;
    }
  sum = 0;
#pragma omp parallel for reduction(+ : sum)
  for (i = 0; i < nints; i++)
    {
      sum = sum + block[i];
    }
#ifdef VERBOSET
  t2 = MPI_Wtime();
  printf("sum of integers %ld %10.3f\n", sum, t2 - t1);
#endif
  free(block);
}

#ifdef STUBS
int omp_get_thread_num(void)
{
  return 0;
}
int omp_get_num_threads(void)
{
  return 1;
}
#endif

void dothreads(int full, char *myname, int myid, int mycolor, int new_id)
{
  int nt, tn;

#pragma omp parallel
  {
    nt = omp_get_num_threads();
    if (nt == 0)
      nt = 1;
#pragma omp critical
    {
      if (nt < 2)
        {
          nt = 1;
          tn = 0;
        }
      else
        {
          tn = omp_get_thread_num();
        }
      if (full == 0)
        {
          if (tn == 0)
            printf(f1236, trim(myname));
        }
      if (full == 1)
        {
          printf(f1235, trim(myname), myid, tn);
        }
      if (full == 2)
        {
          printf(f1234, myid, tn, trim(myname), mycolor, new_id, findcore());
        }
    }
  }
}
