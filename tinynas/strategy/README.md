## Search Strategy module
***
**The search strategy contains a large number of module, mainly complete the search space setting, build super model, budgets, scores, latency and so on  **

- **Strategy Class**

    `build_xxx`: Build sub modules from a cfg dict.

    `budgets`: A property method to get the budgets value setting in cfg.

    `do_compute_nas_score`: Compute a score of a network with specified method.

    `is_satify_budget`:  Determine if a network meets budget. If dissatisfied, we will drop it.

    `get_info_for_evolution`: Build a new model from structure info and generate additional information, such as flops/score ans so on.

