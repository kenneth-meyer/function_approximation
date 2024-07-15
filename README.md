# Function Approximation
Teaching myself how to walk to learn how to run

## What's included in this repository
1. JAX-based code for function approximation in a machine learning setting
2. Classes to be used for polynomial based function approximation
3. Tutorials explaining code + theory
4. Examples comparing different types of function approximation, their strengths and weaknesses, and more

## Ideas for how to make this repo COOL
1. Emphasize communication of results
2. Add an extension that identifies what type of function approximation to use based on user-input. **Motivation**: FEniCS tutorials introduce function spaces very quickly, and very mathematically. Allowing users to use this code without the knowledge of what a Sobolev space is, but in a setting that directs them to use the proper bases/theory to accomplish their goal, would be nice. Explaining function spaces in a user-friendly way would also be nice. Outputting warnings indicating that a user is approximation a given function with a poorly-chosen basis/method is good, but if the user has no idea what those error messages mean, then they will likely be more frustrating than helpful.

## Other ideas
1. Use abstract classes to allow for methods to be agnostic to datatypes; some of my efforts will likely use scipy/numpy only and others will likely use JAX

## Things to do in the future
- make sure this is easily distributable