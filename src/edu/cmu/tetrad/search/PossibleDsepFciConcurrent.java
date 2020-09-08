///////////////////////////////////////////////////////////////////////////////
// For information as to what this class does, see the Javadoc, below.       //
// Copyright (C) 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006,       //
// 2007, 2008, 2009, 2010, 2014, 2015 by Peter Spirtes, Richard Scheines, Joseph   //
// Ramsey, and Clark Glymour.                                                //
//                                                                           //
// This program is free software; you can redistribute it and/or modify      //
// it under the terms of the GNU General Public License as published by      //
// the Free Software Foundation; either version 2 of the License, or         //
// (at your option) any later version.                                       //
//                                                                           //
// This program is distributed in the hope that it will be useful,           //
// but WITHOUT ANY WARRANTY; without even the implied warranty of            //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             //
// GNU General Public License for more details.                              //
//                                                                           //
// You should have received a copy of the GNU General Public License         //
// along with this program; if not, write to the Free Software               //
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA //
///////////////////////////////////////////////////////////////////////////////

package edu.cmu.tetrad.search;

import edu.cmu.tetrad.data.IKnowledge;
import edu.cmu.tetrad.data.Knowledge2;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.util.ChoiceGenerator;
import edu.cmu.tetrad.util.ForkJoinPoolInstance;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

/**
 * This class implements the Possible-D-Sep search step of Spirtes, et al's (1993) FCI algorithm (pp 144-145).
 * Specifically, the methods in this class perform step D. of the algorithm. </p> The algorithm implemented by this
 * class is a bit broader, however, because it allows for the possibility that some pairs of variables have already been
 * compared by a different algorithm. Specifically, if the <code>prevCheck</code> variable is provided in the
 * constructor, then the algorithm pairwise checks every variable in the graph with every variable in v \
 * <code>prevCheck</code> (that is, the unchecked variables). This feature is used by the CIVI algorithm of Danks's
 * "Efficient Inclusion of Novel Variables."
 *
 * @author David Danks
 */
public class PossibleDsepFciConcurrent {

    private Graph graph;
    private IndependenceTest test;

    private SepsetMap sepset;
    private int depth = -1;

    /**
     * The background knowledge.
     */
    private IKnowledge knowledge = new Knowledge2();
    private int maxReachablePathLength = -1;

    /**
     * Concurrency variables
     */
//    private ConcurrentMap<Edge, List<Node>> edgeCondsetMap;
    private int parallelism;
    private double chunk = 5;
    private double factor = 1.25;
//    private ForkJoinPool pool = ForkJoinPoolInstance.getInstance().getPool();
    private ForkJoinPool pool;

    /**
     * Creates a new SepSet and assumes that none of the variables have yet been checked.
     *
     * @param graph The GaSearchGraph on which to work
     * @param test  The IndependenceChecker to use as an oracle
     */
    public PossibleDsepFciConcurrent(Graph graph, IndependenceTest test) {
        if (graph == null) {
            throw new NullPointerException("null GaSearchGraph passed in " +
                    "PossibleDSepSearch constructor!");
        }
        if (test == null) {
            throw new NullPointerException("null IndependenceChecker passed " +
                    "in PossibleDSepSearch " + "constructor!");
        }

        this.graph = graph;
        this.test = test;
        this.sepset = new SepsetMap();

        this.parallelism = Runtime.getRuntime().availableProcessors();
        this.pool = new ForkJoinPool();

        setMaxPathLength(maxReachablePathLength);
    }

    //============================== Public Methods =========================//


    /**
     * Performs pairwise comparisons of each variable in the graph with the variables that have not already been
     * checked. We get the Possible-D-Sep sets for the pair of variables, and we check to see if they are independent
     * conditional on some subset of the union of Possible-D-Sep sets. This method returns the SepSet passed in the
     * constructor (if any), possibly augmented by some edge removals in this step. The GaSearchGraph passed in the
     * constructor is directly changed.
     */
    public SepsetMap search() {

        ConcurrentMap<Edge, List<Node>> edgeCondsetMap = new ConcurrentHashMap<>();

        concurrentSearch(graph, edgeCondsetMap);

        for (Map.Entry<Edge, List<Node>> entry : edgeCondsetMap.entrySet()) {
            Edge edge = entry.getKey();
            List<Node> condSet = entry.getValue();
            Node x = edge.getNode1();
            Node y = edge.getNode2();

            graph.removeEdge(x, y);
            sepset.set(x, y, condSet);
        }

        return sepset;
    }

    public void concurrentSearch(Graph graph, ConcurrentMap<Edge, List<Node>> edgeCondsetMap) {
        final HashMap<Integer,Integer> powerSetSizes = new HashMap<>();
        final ArrayList<Edge> edges = new ArrayList<>(graph.getEdges());
        int sum = 0;
        int psSum;
        for(int i = 0; i < edges.size();i++)
        {
            psSum = 0;
            List<Node> adjNodes = graph.getAdjacentNodes(edges.get(i).getNode1());
            for (int j = 0; j < adjNodes.size(); j++)
            {
                psSum +=  Math.pow(2, graph.getAdjacentNodes(adjNodes.get(j)).size()-1);
            }
            powerSetSizes.put(i, psSum);
            sum += psSum;
        }
        chunk = sum/(parallelism) * factor;

        class searchTask extends RecursiveTask<Boolean> {
            private double chunk;
            private int from;
            private int to;
            public searchTask(double chunk,int from, int to)
            {
                this.chunk = chunk;
                this.from = from;
                this.to = to;
            }
            protected Boolean compute()
            {
                int numTests = 0;
                for(int j = from; j < to; j++)
                {
                    numTests += powerSetSizes.get(j);
                }
                if(to-from <= 2 || numTests <= chunk)
                {
                    System.out.println(Thread.currentThread().getName() + ": " + numTests + " / " + chunk);
                    for(int i = from; i < to; i ++)
                    {
                        final Edge edge = edges.get(i);
                        Node x = edge.getNode1();
                        Node y = edge.getNode2();

                        List<Node> condSet = getSepset(x, y);

                        if (condSet != null) {
                            for (Node n : condSet) {
                                if (!(graph.getAdjacentNodes(n).contains(x) || graph.getAdjacentNodes(n).contains(y))) {
                                    System.out.println("Not adjacent");
                                }
                            }

                            edgeCondsetMap.put(edge, condSet);

                            System.out.println("Removed " + x + "--- " + y + " sepset = " + condSet);
                        }
                    }
                    System.out.println(Thread.currentThread().getName() + ": Finished");
                    return true;
                }
                else
                {
                    List<searchTask> tasks = new ArrayList<>();
                    final int mid = (to+from)/2;
                    searchTask t1 = new searchTask(chunk,from,mid);
                    tasks.add(t1);
                    searchTask t2 = new searchTask(chunk,mid,to);
                    tasks.add(t2);
                    invokeAll(tasks);
                    return true;
                }

            }
        }
        pool.invoke(new searchTask(chunk,0,edges.size()));
        pool.shutdown();
    }

    public List<Node> getSepset(Node node1, Node node2) {
        List<Node> condSet = getCondSet(node1, node2, maxReachablePathLength);

        if (sepset == null) {
            condSet = getCondSet(node2, node1, maxReachablePathLength);
        }

        return condSet;
    }

    private List<Node> getCondSet(Node node1, Node node2, int maxPathLength) {
        final Set<Node> possibleDsepSet = getPossibleDsep(node1, node2, maxPathLength);
        List<Node> possibleDsep = new ArrayList<>(possibleDsepSet);
        boolean noEdgeRequired = getKnowledge().noEdgeRequired(node1.getName(), node2.getName());

        List<Node> possParents = possibleParents(node1, possibleDsep, getKnowledge());

        int _depth = getDepth() == -1 ? 1000 : getDepth();

        for (int d = 0; d <= Math.min(_depth, possParents.size()); d++) {
            ChoiceGenerator cg = new ChoiceGenerator(possParents.size(), d);
            int[] choice;

            while ((choice = cg.next()) != null) {
                List<Node> condSet = GraphUtils.asList(choice, possParents);
                boolean independent = test.isIndependent(node1, node2, condSet);

                if (independent && noEdgeRequired) {
                    return condSet;
                }
            }
        }

        return null;
    }

    /**
     * Removes from the list of nodes any that cannot be parents of x given the background knowledge.
     */
    private List<Node> possibleParents(Node x, List<Node> nodes,
                                       IKnowledge knowledge) {
        List<Node> possibleParents = new LinkedList<>();
        String _x = x.getName();

        for (Node z : nodes) {
            String _z = z.getName();

            if (possibleParentOf(_z, _x, knowledge)) {
                possibleParents.add(z);
            }
        }

        return possibleParents;
    }

    private boolean possibleParentOf(String _z, String _x, IKnowledge bk) {
        return !(bk.isForbidden(_z, _x) || bk.isRequired(_x, _z));
    }

    /**
     * A variable v is in Possible-D-Sep(A,B) iff
     * <pre>
     * 	(i) v != A & v != B
     * 	(ii) there is an undirected path U between A and v such that for every
     * 		 subpath <X,Y,Z> of U either:
     * 		(a) Y is a collider on the subpath, or
     * 		(b) X is adjacent to Z.
     * </pre>
     */
    private Set<Node> getPossibleDsep(Node node1, Node node2, int maxPathLength) {
        Set<Node> dsep = GraphUtils.possibleDsep(node1, node2, graph, maxPathLength);

        dsep.remove(node1);
        dsep.remove(node2);

//        TetradLogger.getInstance().log("details", "Possible-D-Sep(" + node1 + ", " + node2 + ") = " + dsep);

        return dsep;
    }

    public int getDepth() {
        return depth;
    }

    public void setDepth(int depth) {
        if (depth < -1) {
            throw new IllegalArgumentException(
                    "Depth must be -1 (unlimited) or >= 0: " + depth);
        }

        this.depth = depth;
    }

    public IKnowledge getKnowledge() {
        return knowledge;
    }

    public void setKnowledge(IKnowledge knowledge) {
        this.knowledge = knowledge;
    }

    public int getMaxReachablePathLength() {
        return maxReachablePathLength == Integer.MAX_VALUE ? -1 : maxReachablePathLength;
    }

    public void setMaxPathLength(int maxReachablePathLength) {
        if (maxReachablePathLength < -1) {
            throw new IllegalArgumentException("Max path length must be -1 (unlimited) or >= 0: " + maxReachablePathLength);
        }

        this.maxReachablePathLength = maxReachablePathLength == -1 ? Integer.MAX_VALUE : maxReachablePathLength;
    }
}




