## Search Searcher module
Searcher module can be built from a config file and run search process easily, which can execute in local-mode（single-process） or distribute-mode（multi-process). During searching, the master process will cache intermediate results, and export and save the final best structure.
 
- **Searcher Class**

    `__init__`: Setup searching envirioment and then build strategy, sychonizer and population.

    `run`: Execute search loop and export the best structure info.

    `search_loop`: Master process collect and assgin newest population with workers and cache intermediate results.  Then all processes execute search step.

    `search_step`: In search step, all processes pick a random structure info from population and then mutate a new structure info using strategy. If the new network meets the budget and has a high score, we will insert it to the population.

    `export_cache_generation`: cache the search intermediate population every log_freq .
    
    `export_result`: save and export best network structure info.

- **Synchonizer Class**

    Sychonize population information between master and workers using mpi asynchronous communication.



