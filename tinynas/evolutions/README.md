## Population module

Currently, Only a simple population class is implemented, which maintain the Populaiton. The information in Population consists of different ranked list, including Structure_info, ACC, Score, Paramters, FLOPs, Latency, Layers, Stages. More information for the candidate structure could be added freely.

- **Population Class**

    `init_population`: Initialize population parameters and information list.

    `update_population`: Update the individual network information that meets the searched budgets.

    `rank_population`: Rank the Population info list with ACC.

    `merge_shared_data`: Merge different Population info between different threads.

    `export_dict`: Export the whole Population info list to a dict for the searching process.

    `get_individual_info`: Get the individual network information with index.
