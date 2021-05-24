/*
 *  EstimateBetweenness.h
 *
 *  Created on: 13.06.2014
 *      Author: Christian Staudt, Elisabetta Bergamini
 */

#ifndef NETWORKIT_CENTRALITY_ESTIMATE_BETWEENNESS_HPP_
#define NETWORKIT_CENTRALITY_ESTIMATE_BETWEENNESS_HPP_

#include <networkit/centrality/Centrality.hpp>
#include <vector>


namespace NetworKit {

/**
 * @ingroup centrality
 * Estimation of betweenness centrality according to algorithm described in
 * Sanders, Geisberger, Schultes: Better Approximation of Betweenness Centrality.
 * There is no proven theoretical guarantee on the quality of the approximation. However, the algorithm was shown to perform well in practice.
 * If a guarantee is required, use ApproxBetweenness.
 */
class EstimateBetweenness: public Centrality {

public:

    /**
     * The algorithm estimates the betweenness of all nodes, using weighting
     * of the contributions to avoid biased estimation. The run() method takes O(m)
     * time per sample, where  m is the number of edges of the graph.
     *
     * @param	graph		input graph
     * @param	nSamples	 user defined number of samples
     * @param	normalized   normalize centrality values in interval [0,1] ?
     * @param	parallel_flag	if true, run in parallel with additional memory cost z + 3z * t
     * @param   seed            seed used for sampling of nodes
     * @param   sources        vector of the source nodes
     * @param   targets        vector of the target nodes
     * @param   ident          vector of the ident scores for each node in the graph
     * @param   weightedSampling if true, uses the ident vector to perform weighted sampling without replacement from the 'sources' list
     */
     EstimateBetweenness(const Graph& G, count nSamples, bool normalized=false, bool parallel_flag=false, unsigned seed=0, std::vector<size_t> sources={}, std::vector<size_t> targets={}, std::vector<size_t> ident={}, bool weightedSampling=false);

     /**
     * Computes betweenness estimation on the graph passed in constructor.
     */
    void run() override;


private:

    count nSamples;
    bool parallel_flag;
    unsigned seed;
    std::vector<size_t> sources;
    std::vector<size_t> targets;
    std::vector<size_t> ident;
    bool weightedSampling;

    std::vector<node> get_node_samples(const std::vector<size_t>& sources, const std::vector<size_t>& ident, const count nSamples, unsigned seed, bool weightedSampling);

};

} /* namespace NetworKit */

#endif // NETWORKIT_CENTRALITY_ESTIMATE_BETWEENNESS_HPP_
