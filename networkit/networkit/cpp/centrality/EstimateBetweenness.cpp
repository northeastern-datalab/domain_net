/*
 *  EstimateBetweenness.cpp
 *
 *  Created on: 13.06.2014
 *      Author: Christian Staudt, Elisabetta Bergamini
 */


#include <networkit/centrality/EstimateBetweenness.hpp>
#include <networkit/distance/BFS.hpp>
#include <networkit/distance/Dijkstra.hpp>
#include <networkit/distance/SSSP.hpp>
#include <networkit/auxiliary/SignalHandling.hpp>
#include <networkit/auxiliary/Parallelism.hpp>
#include <networkit/graph/GraphTools.hpp>

#include <memory>
#include <omp.h>


#include <iostream>
#include <random>
#include <algorithm>
#include <map>


namespace NetworKit {

EstimateBetweenness::EstimateBetweenness(const Graph& G, count nSamples, bool normalized, bool parallel_flag,
 unsigned seed, std::vector<size_t> sources, std::vector<size_t> targets, std::vector<size_t> ident, bool weightedSampling) : 
 Centrality(G, normalized), nSamples(nSamples), parallel_flag(parallel_flag), seed(seed), sources(sources), targets(targets), ident(ident), weightedSampling(weightedSampling){
}

/**
 * Given a vector of source nodes and an ident vector sample 'nSamples' nodes
 * 
 * Related resource for weighted random sampling: https://stackoverflow.com/questions/1761626/weighted-random-numbers
 * 
 * Returns a vector of the sampled nodes
 */
std::vector<node> EstimateBetweenness::get_node_samples(const std::vector<size_t>& sources, const std::vector<size_t>& ident, const count nSamples, unsigned seed, bool weightedSampling) {
    // There should be a one to one mapping between the sources vector and the ident vector
    assert(sources.size() == ident.size());

    // Vector of node IDs to be returned as the sampled nodes
    std::vector<node> sampledNodes;

    if (sources.size() == nSamples) {
        // Special case when the nSamples equal to the number of nodes in the 'sources' vector
        // No need to perform any random sampling
        sampledNodes = sources;
        return sampledNodes;
    }

    std::mt19937 gen(seed);    
    if (weightedSampling) {
        // Perform weighted random sampling without replacement
        std::vector<size_t> cur_ident_vec(ident);

        while (sampledNodes.size() < nSamples) {
            // Update distribution to the latest 'cur_ident_vec'
            std::discrete_distribution<size_t> dist(cur_ident_vec.begin(), cur_ident_vec.end());

            size_t selected_id = dist(gen);
            sampledNodes.push_back((node)sources[selected_id]);

            // Set 0 to the 'selected_id' index in the cur_ident_vec, because we have already selected that node_id
            cur_ident_vec[selected_id] = 0;
        }
    }
    else {
        // Perform unweighted random sampling
        std::vector<size_t> sources_tmp = sources;
        std::shuffle(sources_tmp.begin(), sources_tmp.end(), gen);
        // Select the first nSamples nodes from target_nodes
        for (count i = 0; i < nSamples; ++i) {
            sampledNodes.push_back((node)sources_tmp[i]);
        }
    }

    return sampledNodes;
}


void EstimateBetweenness::run() {
    hasRun = false;

    Aux::SignalHandler handler;

    // Ensure that the number of samples is less that the possible source nodes
    if (nSamples > sources.size()) {
        std::cerr << "Cannot have more samples than the number of source nodes\n";
        exit(1);
    }

    // Sample 'nSamples' from the list of 'sources' nodes
    // TODO: Track how long the sampling process takes
    std::vector<node> sampledNodes = get_node_samples(sources, ident, nSamples, seed, weightedSampling);
    

    // thread-local scores for efficient parallelism
    count maxThreads = omp_get_max_threads();
    if (!parallel_flag) maxThreads = 1;
    std::vector<std::vector<double> > scorePerThread(maxThreads, std::vector<double>(G.upperNodeIdBound()));


    auto computeDependencies = [&](node s){
        // run single-source shortest path algorithm
        std::unique_ptr<SSSP> sssp;
        if (G.isWeighted()) {
            // Note: Ident has not been implemented for weighted graphs
            sssp = std::make_unique<Dijkstra>(G, s, true, true);
        } else {
            sssp = std::make_unique<BFS>(G, s, true, true, none, ident);
        }
        if (!handler.isRunning()) return;
        sssp->run();
        if (!handler.isRunning()) return;


        // create stack of nodes in non-decreasing order of distance
        auto stack = sssp->getNodesSortedByDistance();

        // compute dependencies and add the contributions to the centrality score
        std::vector<double> dependency(G.upperNodeIdBound(), 0.0);

        std::unordered_set<node> targets_set(targets.begin(), targets.end());
        targets_set.erase(s);
        for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
            node t = *it;

            double coeff;
            if (targets_set.find(t) != targets_set.end()) {
                bigfloat tmp = sssp->numberOfPaths(t);
                double tmp_double;
                tmp.ToDouble(tmp_double);
                coeff = (dependency[t] + 1.0) / tmp_double;
            }
            else {
                bigfloat tmp = sssp->numberOfPaths(t);
                double tmp_double;
                tmp.ToDouble(tmp_double);
                coeff = (dependency[t]) / tmp_double;
            }

            for (node p : sssp->getPredecessors(t)) {
                bigfloat tmp = sssp->numberOfPaths(p);
                double tmp_double;
                tmp.ToDouble(tmp_double);
                dependency[p] += (double(sssp->distance(p)) / sssp->distance(t)) * tmp_double * coeff * ident[t];
            }
            if (t != s) {
                scorePerThread[omp_get_thread_num()][t] += (dependency[t] * ident[s]);
            }

            // // OLD Implementation that does not consider source and target subsets
            // if (t == s){
            //     continue;
            // }
            // for (node p : sssp->getPredecessors(t)) {
            //     // TODO: make weighting factor configurable

            //     // workaround for integer overflow in large graphs
            //     bigfloat tmp = sssp->numberOfPaths(p) / sssp->numberOfPaths(t);
            //     double weight;
            //     tmp.ToDouble(weight);

            //     dependency[p] += (double(sssp->distance(p)) / sssp->distance(t)) * weight * (1 + dependency[t]);
            // }
            // scorePerThread[omp_get_thread_num()][t] += dependency[t];
        }
    };


    #pragma omp parallel for if(parallel_flag)
    for (omp_index i = 0; i < static_cast<omp_index>(sampledNodes.size()); ++i) {
        computeDependencies(sampledNodes[i]);
    }

    if (parallel_flag) {
        scoreData = std::vector<double>(G.upperNodeIdBound(), 0.0);

        // add up all thread-local values
        for (const auto &local : scorePerThread) {
            G.parallelForNodes([&](node v){
                scoreData[v] += local[v];
            });
        }
    } else {
        scoreData.swap(scorePerThread[0]);
    }

    const count n = G.numberOfNodes();
    const count pairs = (n-2) * (n-1);

    // extrapolate
    G.parallelForNodes([&](node u) {
        // scoreData[u] = scoreData[u] * (2 * static_cast<double>(n) / static_cast<double>(nSamples));

        if (normalized) {
            // divide by the number of possible pairs
            scoreData[u] = scoreData[u] / pairs;
        }
    });

    handler.assureRunning();
    hasRun = true;
}


} /* namespace NetworKit */
