# Introduction to JAX

This is an introductory set of notes on JAX and XLA with a focus on machine learning.

These notes assume no prior knowledge of machine learning or accelerator programming. The goal is to focus on practical programming concepts, performance techniques, and the "why" behind JAX's design.

## Table of Contents

1. [Just-In-Time Compilation](01%20Just-In-Time%20Compilation.ipynb): Speed up Python functions using JAX’s `jit` compiler and the XLA backend. Learn when JIT helps and how to get the most out of it.

2. [Automatic Vectorization](02%20Automatic%20Vectorization.ipynb): Replace slow Python loops with batched, parallel computation using `vmap`. Keep your code clean while making it run faster.

3. [Automatic Differentiation](03%20Automatic%20Differentiation.ipynb): Calculate derivatives and gradients with `grad`, `value_and_grad`, and more. Apply them to simple functions or complex, structured inputs.

4. [Debugging](04%20Debugging.ipynb): See what’s happening inside JIT or vectorized code with `jax.debug.print` and breakpoints. Learn techniques for finding and fixing issues in compiled functions.

5. [Pytrees](05%20Pytrees.ipynb): Use JAX’s nested data structures to handle complex inputs and outputs. Map, transform, and even register your own custom types.

6. [Distributed Computing](06%20Distributed%20Computing.ipynb): Run computations efficiently across multiple devices with sharding and parallelism. Explore automatic, explicit, and manual approaches.

7. [Stateful Computation](07%20Stateful%20Computation.ipynb): Keep randomness and model state reproducible while staying pure. Structure your code so it works seamlessly with JAX transformations.

8. [JIT Control Flow and Logic](08%20JIT%20Control%20Flow%20and%20Logic.ipynb): Write conditionals and loops that run efficiently in JIT-compiled functions. Use JAX’s control flow tools to avoid retracing and unrolling.

## Interactivity

Each chapter in these notes is presented as a Jupyter notebook.

To get started with running the notebooks, first install the `uv` package manager, then run `uv sync` to install the required packages.

Then, launch JupyterLab (the notebook interface) with `uv run --with jupyter jupyter lab` or point your favorite editor to the created IPython kernel.

> [!TIP]
> You're encouraged to actively engage with them by experimenting with the code; hands-on exploration is one of the best ways to learn. For even deeper understanding, consider working through concepts using pen, paper, and your preferred editor.

## File Structure

```
├── 01 Just-In-Time Compilation.ipynb
├── 02 Automatic Vectorization.ipynb
├── 03 Automatic Differentiation.ipynb
├── 04 Debugging.ipynb
├── 05 Pytrees.ipynb
├── 06 Distributed Computing.ipynb
├── 07 Stateful Computation.ipynb
├── 08 JIT Control Flow and Logic.ipynb
├── README.md
├── pyproject.toml
└── uv.lock
```

## Further Reading

- https://docs.jax.dev/en/latest/advanced_guide.html
- https://docs.jax.dev/en/latest/user_guides.html
- https://docs.jax.dev/en/latest/changelog.html

## Disclaimer

There may be typos or errors in these notes. This, as most online resources, should not be treated as a single source of truth (especially given the rapid rate of change in ML frameworks) and should rather be an additional resource used in parallel with official JAX documentation and changelogs.

If you spot an issue, feel free to open an issue!
