# Introduction to JAX

This is an introductory set of notes on JAX and XLA with a focus on machine learning.

It does not assume prior machine learning or accelerator programming background.
The goal of the notes are to focus on practical programming concepts, performance techniques, and the "why" behind JAX's design.

## Table of Contents

1. [Just-In-Time Compilation](01%20Just-In-Time%20Compilation.ipynb)
2. [Automatic Vectorization](02%20Automatic%20Vectorization.ipynb)
3. [Automatic Differentiation](03%20Automatic%20Differentiation.ipynb)
4. [Debugging](04%20Debugging.ipynb)
5. [Pytrees](05%20Pytrees.ipynb)
6. [Distributed Computing](06%20Distributed%20Computing.ipynb)
7. [Stateful Computation](07%20Stateful%20Computation.ipynb)
8. [JIT Control Flow and Logic](08%20JIT%20Control%20Flow%20and%20Logic.ipynb)

## Interactivity

Each chapter in these notes is presented as a Jupyter notebook. You're encouraged to actively engage with them by experimenting with the code; hands-on exploration is one of the best ways to learn. For even deeper understanding, consider working through concepts using pen, paper, and your preferred editor.

To get started with running the notebooks, run `uv sync` to install the required packages.

Then, launch JupyterLab (the notebook interface) with `uv run --with jupyter jupyter lab` or point your favorite editor to the created IPython kernel.

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
