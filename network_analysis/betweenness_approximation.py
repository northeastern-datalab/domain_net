import networkx as nx
import networkit as nk
import pandas as  pd
import pickle
import utils
from timeit import default_timer as timer
from decimal import Decimal


import argparse

from tqdm import tqdm

def main(args):
    G = pickle.load(open(args.graph, 'rb'))

    if args.dataframe:
        df = pickle.load(open(args.dataframe, 'rb'))

    min_samples=args.sample_size[0]
    max_samples=args.sample_size[1]
    step_size=args.sample_size[2]

    # Convert graph from networkx to networkit
    G_nk = nk.nxadapter.nx2nk(G)

    df_elapsed_time = pd.DataFrame(columns=['sample_size', 'geisberger2008'])
    df_elapsed_time['sample_size'] = range(min_samples, max_samples+1, step_size)

    # Run LCC
    if args.run_LCC:
        start = timer()
        print('Calculating LCC scores...')
        lcc = nk.centrality.LocalClusteringCoefficient(G_nk, turbo=False).run()
        lcc_scores = lcc.scores()
        print('Finished calculating LCC scores. Elapsed time is', timer()-start, 'seconds')

    # Vary sample size and compute BC for every node using Geisberger2008
    run_times = []
    for sample_size in range(min_samples, max_samples+1, step_size):
        start = timer()
        bc_scores = utils.betweenness.betweeness_approximate(G_nk, num_samples=sample_size, quiet=True)
        elapsed_time = timer()-start
        run_times.append(elapsed_time)

        if args.dataframe:
            df['geisberger_' + str(sample_size)] = bc_scores

        print('Elapsed time for', sample_size, 'samples is:', elapsed_time)
        comps_per_second = (G_nk.numberOfEdges() * sample_size) / elapsed_time
        print('Computations per second:', f"{Decimal(comps_per_second):.2E}", '\n')
    df_elapsed_time['geisberger2008'] = run_times

    # Save runtimes
    df_elapsed_time.to_pickle(args.output_dir + 'approximate_benchmark_runtimes.pickle')

    if args.dataframe:
        # Save df of approximate BC scores benchmark
        df.to_pickle(args.output_dir + 'graph_stats_approximate_benchmark.pickle')


if __name__ == "__main__":
        # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='comparison of BC approximation methods')

    # Output directory where output files and figures are stored
    parser.add_argument('-o', '--output_dir', metavar='output_dir', required=True,
    help='Path to the output directory where output files and figures are stored. \
    Path must terminate with backslash "\\"')

    # Input graph representation of the set of tables
    parser.add_argument('-g', '--graph', metavar='graph', required=True,
    help='Path to the Graph representation of the set of tables')

    # Input graph representation of the set of tables. If not specified the scores are not saved and only the runtime is logged
    parser.add_argument('-df', '--dataframe', metavar='df',
    help='Path to the graph stats dataframe')

    # Denotes the sample sizes to test with geisberger2008 approximation algorithm.
    # Argument is a range of min to max sample size and the step size.
    parser.add_argument('--sample_size', nargs=3, type=int, metavar=('min_sample_size', 'max_sample_size', 'step'),
    default=[5,500,50],
    help='Denotes the sample sizes to test with geisberger2008 approximation algorithm.\
    Argument is a range of min to max sample size and the step size.')

    # If specified runs LCC as well
    parser.add_argument('--run_LCC', action='store_true', 
    help='If specified runs LCC as well')

    # Parse the arguments
    args = parser.parse_args()
   
    print('##### ----- Running network_analysis/betweenness_approximation.py with the following parameters ----- #####\n')

    print('Output directory:', args.output_dir)
    print('Graph path:', args.graph)
    if args.dataframe:
        print('Graph stats dataframe path:', args.dataframe)
    print('Sample sizes to test in the range:',  args.sample_size[0],'-', args.sample_size[1],
        'with step size:', args.sample_size[2])
    if args.run_LCC:
        print('In addition run LCC')
    print('\n\n')
    
    main(args)

