
void fill_zeros(double *to_fill, int size)
{
  for(int k = 0; k < size; ++k)
  {
    to_fill[k] = 0.0;
  }
  return;
}

void fill_ones(double *to_fill, int size)
{
  for(int k = 0; k < size; ++k)
  {
    to_fill[k] = 1.0;
  }
  return;
}

void fill_value(double *to_fill, int size, double value)
{
  for (int k = 0; k < size; ++k)
  {
    to_fill[k] = value;
  }
  return;
}

void fill_cb(double *to_fill, int size, double (*ff)(int))
{
  for (int k = 0; k < size; ++k)
  {
    to_fill[k] = ff(k);
  }
  return;
}
