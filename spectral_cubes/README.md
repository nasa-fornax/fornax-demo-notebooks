# Analytical data search in the cloud: finding jets in JWST spectral cubes

Not all archival search queries can be answered with metadata alone. In this use case, we start from an intimidating user query: "I want to find JWST spectral image cubes containing Fe II emission from jets launched by young stellar objects." In the course of answering this query, we learn to think carefully about the order of operations for coordinate searches on large numbers of targets, to perform parts of this search locally when there are too many targets for archive services to handle, to parallelize our data processing, to load data into memory instead of storage, and to load data from the cloud instead of on-premise storage.

Although this tutorial focuses on JWST spectral cubes, it is intended to teach general strategies that will be useful for answering other data-intensive search queries unrelated to spectral cubes, in any of the Fornax archives. We design some simple algorithms to search for extended emission in spectral cubes, but these algorithms are designed to be replaceable by more sophisticated tools tailored to your science case.

Note: the notebook (emission_search_cubes.md) requires the files in the code_src/ directory in order to run.
