---
layout: default
title: Applications
parent: Swift
grand_parent: Systems
---
# Swift applications

Some optimized versions of common applications are provided for the Swift cluster. Below is a list of how to utilize these applications and the optimizations for Swift. 

## Modules
Many are available as part of the [Modules](./modules) setup.

## TensorFlow

TensorFlow has been built for the AMD architecture on Swift. This was done by using the following two build flags. 

```
-march=znver2
-mtune=znver2
```

This version of TensorFlow can be installed from a wheel file: 
```
pip install --upgrade --no-deps --force-reinstall /nopt/nrel/apps/wheels/tensorflow-2.4.2-cp38-cp38-linux_x86_64-cpu.whl
```

Currently, this wheel is not built with NVIDIA CUDA support for running on GPU. 

**TensorFlow installed on Swift with Conda may be significantly slower than the optimized version** 
