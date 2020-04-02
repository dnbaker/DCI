### DCI: Dynamic Continuous Indexing

Fast, generic implementations of [Dynamic Continuous Indexing](https://arxiv.org/abs/1512.00442) and [Prioritized Dynamic Continuous Indexing](https://arxiv.org/abs/1703.00440) (PDCI).

Additionally, includes matrix hashes for several kinds of dissimilarity measures. These include:

1. L2 distance (both E2LSH and P-stable LSHashing)
2. L1 distance (P-stable)
3. Total Variation Distance (a special case of P-stable L1)
4. Lp distance, 1 < p < 2, using CMS sampling
5. Jensen-Shannon Divergence
6. S2JSD (Jensen-Shannon Metric, the square root of the JSD)
7. Hellinger Distance

These can be used either in a table, such as `dci::hash::LSHTable`, or for DCI.

The [Fast Randomized Projections](https://github.com/dnbaker/frp) project in which this was originally developed also has an
FHTHasher, which computes the projections using the FHT compatible with this.
