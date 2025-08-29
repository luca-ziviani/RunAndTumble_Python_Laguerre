# Run and Tumble model - Laguerre - Python

This repo contains a python code which compute a numeric solution the Run and Tumble equation. 
$$\partial_tf (t,x,v)+ v\cdot\nabla_x f (t,x,v)= \mathcal{M}(v) \int_{\mathbb{R}^d}\Lambda\left( \frac{x\cdot v'}{\lfloor x \rceil}\right) f(t,x,v')dv '-\Lambda\left(  \frac{x\cdot v}{\lfloor x \rceil}\right)f(t,x,v),$$

## How it works:

We consider a interval in position x and an inteval in velocity v.

The local equilibrium is a $\mathcal{M}(v) = \exp(-|v|)$

the function $\psi$ is $sign$.



