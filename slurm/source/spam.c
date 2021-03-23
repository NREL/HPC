/*
 * from Python
 * from spam import *
 * findcore()    : return the core on which a process is running
 * forcecore(c)  : move the calling process to given core
 * p_to_c(pid,c) : move process pid to core c, not currently working on Eagle
*/

/**********
   To copile for use with a C program compile with the line:
   gcc -DCONLY -c spam.c

   EXAMPLE from C:
#include <stdio.h>

void FORCECORE (int *core);
void FINDCORE (int *core);
void P_TO_C (int * pid ,int *core);
main() {
  int i,j;
  FINDCORE(&i);
  printf("on core %d\n",i);
  j=(i+1) % 4;
  printf("requesting core %d\n",j);
  FORCECORE(&j);
  FINDCORE(&i);
  printf("on core %d\n",i);
}

**********/
 
#ifndef CONLY
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#else
#define _GNU_SOURCE
#endif

#include <inttypes.h>
#include <stdio.h>


#include <sys/types.h> 
#include <unistd.h> 
#include <sched.h> 
#include <stdlib.h> 
#include <unistd.h> 
#include <stdio.h> 
#include <assert.h> 

pid_t getpid(void); 

void FORCECORE (int *core) { 
	int bonk; 
	cpu_set_t set; 
	bonk=*core; 
        bonk=abs(bonk) ;
	CPU_ZERO(&set);        // clear cpu mask 
	CPU_SET(bonk, &set);      // set cpu 0 
        if (*core < 0 ){
	 	sched_setaffinity(0, sizeof(cpu_set_t), &set);   
        }else{
	        sched_setaffinity(getpid(), sizeof(cpu_set_t), &set);   
        }
} 

void FINDCORE (int *ic) 
{ 
    ic[0] = sched_getcpu(); 
} 


void P_TO_C (int * pid ,int *core) { 
        cpu_set_t set;
        pid_t apid;
        int i;
        apid=(pid_t)*pid;
        CPU_ZERO(&set);        // clear cpu mask 
        int bonk;
        bonk=*core;
        CPU_SET(bonk, &set);      // set cpu 0 
        i=sched_setaffinity(apid, sizeof(cpu_set_t), &set);
        printf("%d\n",i);
	printf("not working %ld %d call from process\n",(long)apid,bonk);
} 

#ifndef CONLY
static PyObject * findcore(PyObject *self, PyObject *args)
{
    size_t gotit;
    int core;
    FINDCORE(&core);
    gotit=(long)core;
    return PyLong_FromLong((long)gotit);
}

static PyObject * forcecore(PyObject *self, PyObject *args)
{
    size_t gotit;
    long lastbase;
    int core;
    if (!PyArg_ParseTuple(args, "l", &lastbase))
        return NULL;
    core=(int)lastbase;
    FORCECORE(&core);
    gotit=0;
    return PyLong_FromLong((long)gotit);
}


static PyObject * p_to_c(PyObject *self, PyObject *args)
{
    size_t id,p;
    size_t gotit;

    int pid,core;
    if (!PyArg_ParseTuple(args, "ll", &id,&p))
        return NULL;
    pid=(int)id;
    core=(int)p;
    P_TO_C(&pid,&core);
    gotit=0;
    //printf("gotit= %ld\n",(long)gotit);
    return PyLong_FromLong((long)gotit);
}



// Py_INCREF(Py_None);
// return Py_None;

static PyMethodDef SpamMethods[] = {
    {"findcore",  findcore, METH_VARARGS,"Find the core on which calling task is running"},
    {"forcecore",  forcecore, METH_VARARGS,"Force calling task to a core"},
    {"p_to_c",  p_to_c, METH_VARARGS,"Force task to core"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef spammodule = {
    PyModuleDef_HEAD_INIT,
    "spam",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    SpamMethods
};

PyMODINIT_FUNC
PyInit_spam(void)
{
    return PyModule_Create(&spammodule);
}

#endif
