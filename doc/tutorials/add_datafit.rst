.. _how_to_add_custom_datafit:

.. meta::
   :description: Tutorial on creating and implementing a custom datafit in skglm. Step-by-step guide includes deriving gradients, Hessians, and an example with Poisson datafit.

How to Add a Custom Datafit
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Motivated by generalized linear models but not limited to it, ``skglm`` solves problems of the form

.. math::
      \hat{\beta} \in
      \arg\min_{\beta \in \mathbb{R}^p}
      F(X\beta) + \Omega(\beta)
      := \sum_{i=1}^n f_i([X\beta]_i) + \sum_{j=1}^p \Omega_j(\beta_j)
      \ .


Here, :math:`X \in \mathbb{R}^{n \times p}` denotes the design matrix with :math:`n` samples and :math:`p` features,
and :math:`\beta \in \mathbb{R}^p` is the coefficient vector.

skglm can solve any problems of this form with arbitrary smooth datafit :math:`F` and arbitrary penalty :math:`\Omega` whose proximal operator can be evaluated explicitly, by defining two classes: a ``Penalty`` and a ``Datafit``.

They can then be passed to a :class:`~skglm.GeneralizedLinearEstimator`.

.. code-block:: python

   clf = GeneralizedLinearEstimator(
      MyDatafit(),
      MyPenalty(),
   )


A ``Datafit`` is a jitclass that must inherit from the ``BaseDatafit`` class:

.. literalinclude:: ../../skglm/datafits/base.py
   :pyobject: BaseDatafit


To define a custom datafit, you need to inherit from ``BaseDatafit`` class and implement methods required by the targeted solver.
These methods can be found in the solver documentation.
Optionally, overloading the methods with the suffix ``_sparse`` adds support for sparse datasets (CSC matrix).

This tutorial shows how to implement :ref:`Poisson <skglm.datafits.Poisson>` datafit to be fitted with :ref:`ProxNewton <skglm.solvers.ProxNewton>` solver.


A case in point: defining Poisson datafit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, this requires deriving some quantities used by the solvers like the gradient or the Hessian matrix of the datafit.
With :math:`y \in \mathbb{R}^n` the target vector, the Poisson datafit reads

.. math::
    f(X\beta) = \frac{1}{n}\sum_{i=1}^n \exp([X\beta]_i) - y_i[X\beta]_i
    \ .


Let's define some useful quantities to simplify our computations. For :math:`z \in \mathbb{R}^n` and :math:`\beta \in \mathbb{R}^p`,

.. math::
   f(z) = \sum_{i=1}^n f_i(z_i)  \qquad  F(\beta) = f(X\beta)
   \ .


Computing the gradient of :math:`F` and its Hessian matrix yields

.. math::
   \nabla F(\beta) = X^{\top} \underbrace{\nabla f(X\beta)}_"raw grad" \qquad \nabla^2 F(\beta) = X^{\top} \underbrace{\nabla^2 f(X\beta)}_"raw hessian" X
   \ .


Besides, it directly follows that

.. math::
   \nabla f(z) = (f_i^'(z_i))_{1 \leq i \leq n} \qquad \nabla^2 f(z) = "diag"(f_i^('')(z_i))_{1 \leq i \leq n}
   \ .


We can now apply these definitions to the Poisson datafit:

.. math::
    f_i(z_i) = \frac{1}{n} \left(\exp(z_i) - y_iz_i\right)
    \ .


Therefore,

.. math::
   f_i^('')(z_i) = \frac{1}{n}(\exp(z_i) - y_i) \qquad f^''_i(z_i) = \frac{1}{n}\exp(z_i)
   \ .


Computing ``raw_grad`` and ``raw_hessian`` for the Poisson datafit yields

.. math::
   \nabla f(X\beta) = \frac{1}{n}(\exp([X\beta]_i) - y_i)_{1 \leq i \leq n} \qquad \nabla^2 f(X\beta) = \frac{1}{n}"diag"(\exp([X\beta]_i))_{1 \leq i \leq n}
   \ .


Both ``raw_grad`` and ``raw_hessian`` are methods used by the ``ProxNewton`` solver.
But other optimizers require different methods to be implemented. For instance, ``AndersonCD`` uses the ``gradient_scalar`` method:
it is the derivative of the datafit with respect to the :math:`j`-th coordinate of :math:`\beta`.

For the Poisson datafit, this yields

.. math::
    \frac{\partial F(\beta)}{\partial \beta_j} = \frac{1}{n}
      \sum_{i=1}^n X_{i,j} \left(
         \exp([X\beta]_i) - y
      \right)
      \ .


When implementing these quantities in the ``Poisson`` datafit class, this gives:

.. literalinclude:: ../../skglm/datafits/single_task.py
   :pyobject: Poisson


Note that we have not initialized any quantities in the ``initialize`` method.
Usually, it serves to compute datafit attributes specific to a dataset ``X, y`` for computational efficiency, for example the computation of ``X.T @ y`` in :ref:`Quadratic <skglm.datafits.Quadratic>` datafit.
