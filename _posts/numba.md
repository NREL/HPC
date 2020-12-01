---
title: Speeding up Python Code with Numba
postdate: December 01, 2020
layout: post
author: Ethan Young
description: Numba is a just in time (JIT) compiler for Python and NumPy code. From the official website, "Numba-compiled numerical algorithms in Python can approach the speeds of C or FORTRAN." 
---

# Speeding up Python Code with Numba

Numba is a just in time (JIT) compiler for Python and NumPy code. From their official website:

    Numba translates Python functions to optimized machine code at runtime using the industry-standard LLVM compiler library. Numba-compiled numerical algorithms in Python can approach the speeds of C or FORTRAN.

In this getting-started guide, we build a test environment on Eagle, test the performance of a Numba-compiled function using the `@jit` decorator, and discuss what sorts of applications will see performance improvements.