**Goal**: A pytorch approach to create DCM models.

Simple and Naive(?) idea:
1. Create observation and evolution functions
2. ??? - something with autograd
3. Profit

1. How to represent parameters - and (they are Bayesian) their covariance?
   * They are all gaussian - a vector for the several sets?
   * Dictionaries?
   * Classes?

* How to deal with options?

Probably good idea:
1. A DCM preparation class
2. A evolution class
3. A observation class (usually deterministic - not necessarily)

A simulation process is also needed - class?