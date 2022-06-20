My BenchOpt Benchmark
=====================
|Build Status| |Python 3.6+|

BenchOpt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to solver of TV-1D regularised regression problem:

$$\\boldsymbol{u} \\in \\underset{\\boldsymbol{u} \\in \\mathbb{R}^{p}}{\\mathrm{argmin}} F(\\boldsymbol{y}, A \\boldsymbol{u}) + G(D\\boldsymbol{u})$$


$$G(D\\boldsymbol{u}) = \\lambda \\| D \\boldsymbol{u} \\|_{1} = \\lambda \\| \\boldsymbol{u} \\|_{TV} = \\lambda \\sum\\limits_{k = 0}^{p-1} \\vert u_{k+1} - u_{k} \\vert $$


- $\\boldsymbol{y} \\in \\mathbb{R}^{n}$ is observation as target vector
- $A \\in \\mathbb{R}^{n \\times p}$ is a designed operator as an amplifier.
- $\\lambda > 0$ is a regularization hyperparameter.
- $F(\\cdot)$ is a loss function, like quadratic loss, $F(y, x) = \\frac{1}{2} \\vert y - x \\vert_2^2$, or Huber loss $F(y, x) = h_{\\delta} (y - x)$ defined by


$$   
h_{\\delta}(t) = \\begin{cases} \\frac{1}{2} t^2 & \\textbf{ if } \\vert t \\vert \\le \\delta \\\\ \\delta \\vert t \\vert - \\frac{1}{2} \\delta^2 & \\textbf{ otherwise} \\end{cases}
$$


where n (or n_samples) stands for the number of samples, p (or n_features) stands for the number of features.



Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/benchopt/benchmark_tv_1d
   $ benchopt run benchmark_tv_1d 

Apart from the problem, options can be passed to `benchopt run`, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_tv_1d --config benchmark_tv_1d/example_config.yml


Use `benchopt run -h` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/benchopt/benchmark_tv_1d/workflows/Tests/badge.svg
   :target: https://github.com/benchopt/benchmark_tv_1d/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
