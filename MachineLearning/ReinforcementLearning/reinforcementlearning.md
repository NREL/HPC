---
layout: default
title: Reinforcement Learning   
parent: Machine Learning
---

# Reinforcement Learning with Ray/RLlib

Reinforcement learning algorithms are notorious for the amount of data they need to collect in order to perform adequate agent training. The more data collected, the better the training will be. However, we also need to collect massive amounts of data in reasonable time. That is where RLlib can assist us. 

RLlib is an open-source library for reinforcement learning that offers both high scalability and a unified API for a variety of applications ([source](https://docs.ray.io/en/master/rllib.html)). It supports all known deep learning frameworks such as Tensorflow, Pytorch, although most parts are framework-agnostic and can be used by either one.
