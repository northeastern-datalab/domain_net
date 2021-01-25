#!/usr/bin/env python3

import networkit as nk
import random
import unittest

class TestTraversal(unittest.TestCase):
	def getSmallGraph(self, weighted=False, directed=False):
		G = nk.Graph(4, weighted, directed)
		G.addEdge(0, 1, 1.0)
		G.addEdge(0, 2, 2.0)
		G.addEdge(3, 1, 4.0)
		G.addEdge(3, 2, 5.0)
		G.addEdge(1, 2, 3.0)

		return G

	def generateRandomWeights(self, G):
		if not G.isWeighted():
			G = nk.graph.Traversal.toWeighted(G)
		G.forEdges(lambda u, v, w, eid: G.setWeight(u, v, random.random()))

		return G

	def testBFSfrom(self):
		n = 200
		p = 0.15

		def doBFS(G, sources):
			visited = [False for _ in range(n)]
			sequence = []
			edgeSequence = []
			queue = []

			for source in sources:
				queue.append(source)
				visited[source] = True

			while len(queue) > 0:
				u = queue.pop(0)
				sequence.append(u)
				for v in G.iterNeighbors(u):
					if visited[v] == False:
						queue.append(v)
						visited[v] = True
						edgeSequence.append((u, v))

			return sequence, edgeSequence

		randNodes = [x for x in range(n)]

		for seed in range(1, 4):
			nk.setSeed(seed, False)
			random.seed(seed)
			random.shuffle(randNodes)
			for directed in [False, True]:
				G = nk.generators.ErdosRenyiGenerator(n, p, directed).generate()
				for i in range(1, n + 1):
					sources = randNodes[:i]
					sequence, _ = doBFS(G, sources)

					result = []
					nk.graph.Traversal.BFSfrom(G, sources, lambda u, d: result.append(u))
					self.assertListEqual(sequence, result)

					sources = randNodes[i - 1]
					_, edgeSequence = doBFS(G, [sources])

					result = []
					nk.graph.Traversal.BFSEdgesFrom(
						G, sources, lambda u, v, w, eid: result.append((u, v)))
					self.assertListEqual(edgeSequence, result)

	def testDFSfrom(self):
		n = 200
		p = 0.15

		def doDFS(G, source):
			visited = [False for _ in range(n)]
			sequence = []
			edgeSequence = []
			visited[source] = 1
			stack = [source]

			while len(stack) > 0:
				u = stack.pop()
				sequence.append(u)
				for v in G.iterNeighbors(u):
					if visited[v] == False:
						stack.append(v)
						visited[v] = True
						edgeSequence.append((u, v))

			return sequence, edgeSequence

		for seed in range(1, 4):
			nk.setSeed(seed, False)
			for directed in [False, True]:
				G = nk.generators.ErdosRenyiGenerator(n, p, directed).generate()
				for source in range(n):
					sequence, edgeSequence = doDFS(G, source)

					result = []
					nk.graph.Traversal.DFSfrom(G, source, lambda u: result.append(u))
					self.assertListEqual(sequence, result)

					result = []
					nk.graph.Traversal.DFSEdgesFrom(
						G, source, lambda u, v, w, eid: result.append((u, v)))
					self.assertListEqual(edgeSequence, result)

if __name__ == "__main__":
	unittest.main()
