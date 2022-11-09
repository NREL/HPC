
#include <stdio.h>

#include "silly.h"

const int SIZE = 100;

void println(const char* str)
{
  printf("%s\n", str);
  return;
}

void print_status(int fails, double value)
{
  printf("Array has %d spots different from %g.\n", fails, value);
  return;
}

double fill(int k)
{
  return (double)k;
}

int check_array(const double *array, int size, double value)
{
  int fails = 0;
  for (int k = 0; k < size; ++k)
  {
    if (array[k] != value)
      ++fails;
    else
      continue;
  }
  return fails;
}

int check_cb(const double *array, int size)
{
  int fails = 0;
  for (int k = 0; k < size; ++k)
  {
    if (array[k] != (double) k)
      ++fails;
    else
      continue;
  }
  return fails;
}


int main(int argc, char **argv)
{

  int errors = 0;
  double pi = 3.14159265358979323;

  int fails;
  double my_array[SIZE];

  println("Filling with zeros...");
  fill_zeros(my_array, SIZE);
  println("Checking array...");
  fails = check_array(my_array, SIZE, 0.0);
  // print_status(fails, 0.0);

  errors += fails;

  println("Filling with ones...");
  fill_ones(my_array, SIZE);
  println("Checking array...");
  fails = check_array(my_array, SIZE, 1.0);
  // print_status(fails, 1.0);

  errors += fails;

  println("Filling with pi...");
  fill_value(my_array, SIZE, pi);
  println("Checking array...");
  fails = check_array(my_array, SIZE, pi);
  // print_status(fails, pi);

  errors += fails;

  println("Filling with function values...");
  fill_cb(my_array, SIZE, fill);
  println("Checking array...");
  fails = check_cb(my_array, SIZE);

  errors += fails;

  println("Done.");

  if (errors > 0)
  {
    printf("IT'S BROKEN!\n");
  }
  else
  {
    printf("IT'S WORKING!\n");
  }

  return 0;
}
