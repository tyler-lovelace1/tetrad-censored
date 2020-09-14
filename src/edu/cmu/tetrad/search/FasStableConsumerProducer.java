package edu.cmu.tetrad.search;

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

import edu.cmu.tetrad.data.ContinuousVariable;
import edu.cmu.tetrad.data.IKnowledge;
import edu.cmu.tetrad.data.Knowledge2;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.util.ChoiceGenerator;
import edu.cmu.tetrad.util.TetradLogger;
import org.apache.commons.lang3.RandomStringUtils;

import java.io.PrintStream;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.*;
import java.util.concurrent.*;

/**
 * Implements the "fast adjacency search" used in several causal algorithm in this package. In the fast adjacency
 * search, at a given stage of the search, an edge X*-*Y is removed from the graph if X _||_ Y | S, where S is a subset
 * of size d either of adj(X) or of adj(Y), where d is the depth of the search. The fast adjacency search performs this
 * procedure for each pair of adjacent edges in the graph and for each depth d = 0, 1, 2, ..., d1, where d1 is either
 * the maximum depth or else the first such depth at which no edges can be removed. The interpretation of this adjacency
 * search is different for different algorithm, depending on the assumptions of the algorithm. A mapping from {x, y} to
 * S({x, y}) is returned for edges x *-* y that have been removed.
 * </p>
 * This variant uses the PC-Stable modification, calculating independencies in parallel within each depth.
 * It uses a slightly different algorithm from FasStableConcurrent, probably better.
 *
 * @author Joseph Ramsey.
 */
public class FasStableConsumerProducer implements IFas {

    /**
     * The independence test. This should be appropriate to the types
     */
    private IndependenceTest test;

    /**
     * Specification of which edges are forbidden or required.
     */
    private IKnowledge knowledge = new Knowledge2();

    /**
     * The maximum number of variables conditioned on in any conditional independence test. If the depth is -1, it will
     * be taken to be the maximum value, which is 1000. Otherwise, it should be set to a non-negative integer.
     */
    private int depth = 1000;

    /**
     * The number of independence tests.
     */
    private int numIndependenceTests;


    private TetradLogger logger = TetradLogger.getInstance();

    /**
     * The ns found during the search.
     */
    private SepsetMap sepsets = new SepsetMap();

    /**
     * The depth 0 graph, specified initially.
     */
    private Graph initialGraph;

    // Number formatter.
    private NumberFormat nf = new DecimalFormat("0.00E0");

    /**
     * Set to true if verbose output is desired.
     */
    private boolean verbose = false;

    // The concurrency pool.
//    private ForkJoinPool pool = new ForkJoinPool();
//    private ForkJoinPool pool = ForkJoinPoolInstance.getInstance().getPool();
    private ExecutorService executorService;
    /**
     * Where verbose output is sent.
     */
    private PrintStream out = System.out;

    private int parallelism = Runtime.getRuntime().availableProcessors();

    private double factor = 1.25;

    private int chunk = 100;

    private boolean recordSepsets = true;

    //==========================CONSTRUCTORS=============================//

    /**
     * Constructs a new FastAdjacencySearch.    wd
     */
    public FasStableConsumerProducer(IndependenceTest test) {
        this.test = test;
        this.executorService = Executors.newFixedThreadPool(parallelism+1);
    }

    /**
     * Constructs a new FastAdjacencySearch.
     */
    public FasStableConsumerProducer(Graph initialGraph, IndependenceTest test) {
        this.test = test;
        this.initialGraph = initialGraph;
        this.executorService = Executors.newFixedThreadPool(parallelism+1);
    }

    //==========================PUBLIC METHODS===========================//

    /**
     * Discovers all adjacencies in data.  The procedure is to remove edges in the graph which connect pairs of
     * variables which are independent conditional on some other set of variables in the graph (the "sepset"). These are
     * removed in tiers.  First, edges which are independent conditional on zero other variables are removed, then edges
     * which are independent conditional on one other variable are removed, then two, then three, and so on, until no
     * more edges can be removed from the graph.  The edges which remain in the graph after this procedure are the
     * adjacencies in the data.
     *
     * @return a SepSet, which indicates which variables are independent conditional on which other variables
     */
    public Graph search() {
        this.logger.log("info", "Starting Fast Adjacency Search.");
        long beginTime = System.currentTimeMillis();

        // The search graph. It is assumed going in that all of the true adjacencies of x are in this graph for every node
        // x. It is hoped (i.e. true in the large sample limit) that true adjacencies are never removed.
        Graph graph = new EdgeListGraphSingleConnections(test.getVariables());

        sepsets = new SepsetMap();

        //this is bad when starting from init graph --AJ
        sepsets.setReturnEmptyIfNotSet(true);

        int _depth = depth;

        if (_depth == -1) {
            _depth = 1000;
        }


        Map<Node, Set<Node>> adjacencies = new ConcurrentSkipListMap<>();
        List<Node> nodes = graph.getNodes();

        for (Node node : nodes) {
            adjacencies.put(node, new HashSet<Node>());
        }


        for (int d = 0; d <= _depth; d++) {
            boolean more;

            if (d == 0) {
                more = searchAtDepth0(nodes, test, adjacencies);
            } else {
                more = searchAtDepth(nodes, test, adjacencies, d);
            }

            if (!more) {
                break;
            }
        }

        if (verbose) {
            out.println("Finished with search, constructing Graph...");
        }

        for (int i = 0; i < nodes.size(); i++) {
            for (int j = i + 1; j < nodes.size(); j++) {
                Node x = nodes.get(i);
                Node y = nodes.get(j);

                if (adjacencies.get(x).contains(y)) {
                    graph.addUndirectedEdge(x, y);
                }
            }
        }

        if (verbose) {
            out.println("Finished constructing Graph.");
        }

        if (verbose) {
            this.logger.log("info", "Finishing Fast Adjacency Search.");
        }

        long endTime = System.currentTimeMillis();
        System.out.println("FAS ELAPSED TIME: " + (endTime - beginTime)/1000 + "s");

        executorService.shutdown();
        return graph;
    }

    @Override
    public Graph search(List<Node> nodes) {
        return null;
    }

    @Override
    public long getElapsedTime() {
        return 0;
    }

    public int getDepth() {
        return depth;
    }

    public void setDepth(int depth) {
        if (depth < -1) {
            throw new IllegalArgumentException(
                    "Depth must be -1 (unlimited) or >= 0.");
        }

        this.depth = depth;
    }

    @Override
    public boolean isAggressivelyPreventCycles() {
        return false;
    }

    @Override
    public void setAggressivelyPreventCycles(boolean aggressivelyPreventCycles) {

    }

    @Override
    public IndependenceTest getIndependenceTest() {
        return test;
    }

    public IKnowledge getKnowledge() {
        return knowledge;
    }

    public void setKnowledge(IKnowledge knowledge) {
        if (knowledge == null) {
            throw new NullPointerException("Cannot set knowledge to null");
        }
        this.knowledge = knowledge;
    }

    //==============================PRIVATE METHODS======================/

    class IndependenceTask {
        public Node x;
        public Node y;
        public List<Node> z;

        public IndependenceTask(Node x, Node y, List<Node> z) {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        @Override
        public boolean equals(Object o) {
            if (o == null) {
                return false;
            } else if (!(o instanceof IndependenceTask)) {
                return false;
            } else if ( ((IndependenceTask) o).x != this.x) {
                return false;
            } else if ( ((IndependenceTask) o).y != this.y) {
                return false;
            } else if ( ((IndependenceTask) o).z != this.z) {
                return false;
            }
            return true;
        }

        @Override
        public String toString() {
            return String.format("(" + x + ", " + y + ") | " + z);
        }
    }

    private class Broker {
        public ArrayBlockingQueue<IndependenceTask> queue = new ArrayBlockingQueue<>(1000000);

        public void put(IndependenceTask task) throws InterruptedException {
            queue.put(task);
        }

        public IndependenceTask get() throws InterruptedException {
            return queue.poll(1, TimeUnit.SECONDS);
        }
    }

    private class ProducerDepth0 implements Runnable {
        private Broker broker;
        private List<Node> nodes;
        private IndependenceTask poisonPill;

        public ProducerDepth0(Broker broker, final List<Node> nodes, final IndependenceTask poisonPill) {
            this.broker = broker;
            this.nodes = nodes;
            this.poisonPill = poisonPill;
        }

        @Override
        public void run() {
            System.out.println("\t" + Thread.currentThread().getName() + ": ProducerDepth0 Start");
            try {
                final List<Node> empty = Collections.emptyList();
                for (int i = 0; i < nodes.size(); i++) {

                    Node x = nodes.get(i);

                    for (int j = i+1; j < nodes.size(); j++) {

                        Node y = nodes.get(j);

                        if (initialGraph != null) {
                            Node x2 = initialGraph.getNode(x.getName());
                            Node y2 = initialGraph.getNode(y.getName());

                            if (!initialGraph.isAdjacentTo(x2, y2)) {
                                continue;
                            }
                        }

//                        System.out.println("submitting task:\t" + x + " _||_ " + y);
                        broker.put(new IndependenceTask(x, y, empty));
                    }
                }
                broker.put(poisonPill);
                broker.put(poisonPill);
                System.out.println("\t" + Thread.currentThread().getName() + ": ProducerDepth0 Finish");
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    private class ConsumerDepth0 implements Runnable {
        private Broker broker;
        private List<Node> nodes;
        private IndependenceTest test;
        private Map<Node, Set<Node>> adjacencies;
        private IndependenceTask poisonPill;

        public ConsumerDepth0(Broker broker, final List<Node> nodes, final IndependenceTest test,
                              final Map<Node, Set<Node>> adjacencies, final IndependenceTask poisonPill) {
            this.broker = broker;
            this.nodes = nodes;
            this.test = test;
            this.adjacencies = adjacencies;
            this.poisonPill = poisonPill;
        }

        @Override
        public void run() {
            System.out.println("\t" + Thread.currentThread().getName() + ": ConsumerDepth0 Start");
            try {
                boolean independent;
                IndependenceTask task = broker.get();

                while (task != poisonPill) {

                    if (task == null) {
                        task = broker.get();
                        continue;
                    }

//                    System.out.println("Test for " + task.x + " _||_ " + task.y);

                    try {
                        independent = test.isIndependent(task.x, task.y, task.z);
                    } catch (Exception e) {
                        e.printStackTrace();
                        independent = true;
                    }

//                    System.out.println("Test for " + task.x + " _||_ " + task.y + " returns " + independent);

                    numIndependenceTests++;

                    boolean noEdgeRequired =
                            knowledge.noEdgeRequired(task.x.getName(), task.y.getName());

                    if (independent && noEdgeRequired) {
                        if (recordSepsets && !sepsets.isReturnEmptyIfNotSet()) {
                            getSepsets().set(task.x, task.y, task.z);
                        }

                        if (verbose) {
                            TetradLogger.getInstance().forceLogMessage(SearchLogUtils.independenceFact(task.x, task.y, task.z) + " p = " +
                                    nf.format(test.getPValue()));
                            out.println(SearchLogUtils.independenceFact(task.x, task.y, task.z) + " p = " +
                                    nf.format(test.getPValue()));
                        }
                    } else if (!forbiddenEdge(task.x, task.y)) {
                        adjacencies.get(task.x).add(task.y);
                        adjacencies.get(task.y).add(task.x);

//                        System.out.println("Removed:\t" + task.x + ", " + task.y);
                    }

                    task = broker.get();
                }
                broker.put(poisonPill);
                broker.put(poisonPill);
                System.out.println("\t" + Thread.currentThread().getName() + ": ConsumerDepth0 Finish");
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    private boolean searchAtDepth0(final List<Node> nodes, final IndependenceTest test, final Map<Node, Set<Node>> adjacencies) {
        if (verbose) {
            out.println("Searching at depth 0.");
            System.out.println("Searching at depth 0.");
        }

        Broker broker = new Broker();
        List<Future> status = new ArrayList<>();

        System.out.println("Depth 0:");

        Node x = new ContinuousVariable(RandomStringUtils.randomAlphabetic(15));
        Node y = new ContinuousVariable(RandomStringUtils.randomAlphabetic(15));

        IndependenceTask poisonPill = new IndependenceTask(x, y, new ArrayList<>());

        System.out.println("PoisonPill: " + poisonPill);

        try {
            status.add(executorService.submit(new ProducerDepth0(broker, nodes, poisonPill)));
            for (int i = 0; i < parallelism; i++) {
                status.add(executorService.submit(new ConsumerDepth0(broker, nodes, test, adjacencies, poisonPill)));
            }

            for (int i = 0; i < parallelism+1; i++) {
                status.get(i).get();
            }

        } catch (Exception e) {
            e.printStackTrace();
        }

//        System.out.println(adjacencies);
        return freeDegree(nodes, adjacencies) > 0;

    }

    private boolean forbiddenEdge(Node x, Node y) {
        String name1 = x.getName();
        String name2 = y.getName();

        if (knowledge.isForbidden(name1, name2) &&
                knowledge.isForbidden(name2, name1)) {
            if (verbose) {
                this.logger.log("edgeRemoved", "Removed " + Edges.undirectedEdge(x, y) + " because it was " +
                        "forbidden by background knowledge.");
            }

            return true;
        }

        return false;
    }

    private int freeDegree(List<Node> nodes, Map<Node, Set<Node>> adjacencies) {
        int max = 0;

        for (Node x : nodes) {
            Set<Node> opposites = adjacencies.get(x);

            for (Node y : opposites) {
                Set<Node> adjx = new HashSet<>(opposites);
                adjx.remove(y);

                if (adjx.size() > max) {
                    max = adjx.size();
                }
            }
        }

        return max;
    }

//    private boolean freeDegreeGreaterThanDepth(Map<Node, Set<Node>> adjacencies, int depth) {
//        for (Node x : adjacencies.keySet()) {
//            Set<Node> opposites = adjacencies.get(x);
//
//            if (opposites.size() - 1 > depth) {
//                return true;
//            }
//        }
//
//        return false;
//    }

    private class ProducerDepth implements Runnable {
        private Broker broker;
        private List<Node> nodes;
        private Map<Node, Set<Node>> adjacenciesCopy;
        private int depth;
        private IndependenceTask poisonPill;

        public ProducerDepth(Broker broker, final List<Node> nodes, final Map<Node, Set<Node>> adjacenciesCopy,
                             final int depth, final IndependenceTask poisonPill) {
            this.broker = broker;
            this.nodes = nodes;
            this.adjacenciesCopy = adjacenciesCopy;
            this.depth = depth;
            this.poisonPill = poisonPill;
        }

        @Override
        public void run() {
            System.out.println("\t" + Thread.currentThread().getName() + ": ProducerDepth" + depth + " Start");
            try {
                for (Node x : nodes) {

                    List<Node> adjx = new ArrayList<>(adjacenciesCopy.get(x));

                    for (Node y : adjx) {
                        List<Node> _adjx = new ArrayList<>(adjx);

                        _adjx.remove(y);
                        List<Node> ppx = possibleParents(x, _adjx, knowledge);

                        if (ppx.size() >= depth) {
                            ChoiceGenerator cg = new ChoiceGenerator(ppx.size(), depth);
                            int[] choice;

                            while ((choice = cg.next()) != null) {
                                List<Node> condSet = GraphUtils.asList(choice, ppx);

//                                System.out.println("submitting task:\t" + x + " _||_ " + y + " | " + condSet);
                                broker.put(new IndependenceTask(x, y, condSet));
                            }
                        }
                    }
                }
                broker.put(poisonPill);
                broker.put(poisonPill);
                System.out.println("\t" + Thread.currentThread().getName() + ": ProducerDepth" + depth + " Finish");
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    private class ConsumerDepth implements Runnable {
        private Broker broker;
        private List<Node> nodes;
        private IndependenceTest test;
        private Map<Node, Set<Node>> adjacencies;
        private IndependenceTask poisonPill;

        public ConsumerDepth(Broker broker, final List<Node> nodes, final IndependenceTest test,
                             final Map<Node, Set<Node>> adjacencies, final IndependenceTask poisonPill) {
            this.broker = broker;
            this.nodes = nodes;
            this.test = test;
            this.adjacencies = adjacencies;
            this.poisonPill = poisonPill;
        }

        @Override
        public void run() {
            try {
                boolean independent;
                IndependenceTask task = broker.get();
                int d;
                if (task != null) {
                    d = task.z.size();
                } else {
                    d = 0;
                }
                System.out.println("\t" + Thread.currentThread().getName() + ": ConsumerDepth" + d + " Start");

                while (task != poisonPill) {

                    if (task == null) {
                        task = broker.get();
                        continue;
                    }

//                    System.out.println("Test for " + task.x + " _||_ " + task.y);

                    if (adjacencies.get(task.x).contains(task.y) & adjacencies.get(task.y).contains(task.x)) {

                        try {
                            independent = test.isIndependent(task.x, task.y, task.z);
                        } catch (Exception e) {
                            e.printStackTrace();
                            independent = true;
                        }

//                        System.out.println("Test for " + task.x + " _||_ " + task.y + " | " + task.z + " : " + independent);

                        boolean noEdgeRequired =
                                knowledge.noEdgeRequired(task.x.getName(), task.y.getName());

                        if (independent && noEdgeRequired) {
                            adjacencies.get(task.x).remove(task.y);
                            adjacencies.get(task.y).remove(task.x);

                            if (recordSepsets) {
                                getSepsets().set(task.x, task.y, task.z);
                            }

                            if (verbose) {
                                TetradLogger.getInstance().forceLogMessage(
                                        SearchLogUtils.independenceFact(task.x, task.y, task.z) + " p = " +
                                                nf.format(test.getPValue()));
                                out.println(SearchLogUtils.independenceFact(task.x, task.y, task.z) + " p = " +
                                        nf.format(test.getPValue()));
                            }
                        }
                    }
                    task = broker.get();
                }
                broker.put(poisonPill);
                broker.put(poisonPill);
                System.out.println("\t" + Thread.currentThread().getName() + ": ConsumerDepth" + d + " Finish");
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    private boolean searchAtDepth(final List<Node> nodes, final IndependenceTest test, final Map<Node, Set<Node>> adjacencies,
                                  final int depth) {

        if (verbose) {
            out.println("Searching at depth " + depth);
            System.out.println("Searching at depth " + depth);
        }

        Broker broker = new Broker();
        List<Future> status = new ArrayList<>();
        final Map<Node, Set<Node>> adjacenciesCopy = new HashMap<>();

        Node x = new ContinuousVariable(RandomStringUtils.randomAlphabetic(15));
        Node y = new ContinuousVariable(RandomStringUtils.randomAlphabetic(15));

        IndependenceTask poisonPill = new IndependenceTask(x, y, new ArrayList<>());

        final Map<Node, Integer> testCounts = new HashMap<>();
        int sum = 0;
//        int remainingNodes = 0;
        int testSum;
        for (Node node : adjacencies.keySet()) {
            adjacenciesCopy.put(node, new HashSet<>(adjacencies.get(node)));
            Set<Node> adj = adjacencies.get(node);
//            if (adj.size() >= depth) {
//                remainingNodes += 1;
//            }
            testSum = choose(adj.size(), depth);
            testCounts.put(node, testSum);
            sum += testSum;
        }

//        sum = Math.max(sum, nodes.size());

        System.out.println("Depth " + depth + ":");
        System.out.println("Number of Tests: " + sum);
        System.out.println("PoisonPill: " + poisonPill);

        try {
            status.add(executorService.submit(new ProducerDepth(broker, nodes, adjacenciesCopy, depth, poisonPill)));
            for (int i = 0; i < parallelism; i++) {
                status.add(executorService.submit(new ConsumerDepth(broker, nodes, test, adjacencies, poisonPill)));
            }

            for (int i = 0; i < parallelism+1; i++) {
                status.get(i).get();
            }

        } catch (Exception e) {
            e.printStackTrace();
        }

        return freeDegree(nodes, adjacencies) > depth;

//        chunk = (int) Math.ceil(sum / (double) Math.min(parallelism, remainingNodes) * factor);
//
//        class DepthTask extends RecursiveTask<Boolean> {
//            private int chunk;
//            private int from;
//            private int to;
//
//            public DepthTask(int chunk, int from, int to) {
//                this.chunk = chunk;
//                this.from = from;
//                this.to = to;
//            }
//
//            @Override
//            protected Boolean compute() {
//                long testEdges = 0;
//                for (int i = from; i < to; i++) {
//                    testEdges += testCounts.get(nodes.get(i));
//                }
//
//                if (to - from <= 2 || testEdges <= chunk) {
//                    System.out.println(Thread.currentThread().getName() + ": " + testEdges + " / " + chunk);
//                    for (int i = from; i < to; i++) {
//                        if (verbose) {
//                            if ((i + 1) % 1000 == 0) System.out.println("i = " + (i + 1));
//                        }
//
//                        Node x = nodes.get(i);
//
//                        List<Node> adjx = new ArrayList<>(adjacenciesCopy.get(x));
//
//                        EDGE:
//                        for (Node y : adjx) {
//                            List<Node> _adjx = new ArrayList<>(adjx);
//
//                            _adjx.remove(y);
//                            List<Node> ppx = possibleParents(x, _adjx, knowledge);
//
//                            if (ppx.size() >= depth) {
//                                ChoiceGenerator cg = new ChoiceGenerator(ppx.size(), depth);
//                                int[] choice;
//
//                                COND:
//                                while ((choice = cg.next()) != null) {
//                                    List<Node> condSet = GraphUtils.asList(choice, ppx);
//
//                                    boolean independent;
//
//                                    try {
//                                        numIndependenceTests++;
//                                        independent = test.isIndependent(x, y, condSet);
//                                    } catch (Exception e) {
//                                        independent = false;
//                                    }
//
//                                    boolean noEdgeRequired =
//                                            knowledge.noEdgeRequired(x.getName(), y.getName());
//
//                                    if (independent && noEdgeRequired) {
//                                        adjacencies.get(x).remove(y);
//                                        adjacencies.get(y).remove(x);
//
//                                        if (recordSepsets) {
//                                            getSepsets().set(x, y, condSet);
//                                        }
//
//
//                                        if (verbose) {
//                                            TetradLogger.getInstance().forceLogMessage(
//                                                    SearchLogUtils.independenceFact(x, y, condSet) + " p = " +
//                                                    nf.format(test.getPValue()));
//                                            out.println(SearchLogUtils.independenceFact(x, y, condSet) + " p = " +
//                                                    nf.format(test.getPValue()));
//                                        }
//
//                                        continue EDGE;
//                                    }
//                                }
//                            }
//                        }
//                    }
//
//                    System.out.println(Thread.currentThread().getName() + ": Finished");
//                    return true;
//                } else {
//                    final int mid = (to + from) / 2;
//
//                    List<DepthTask> tasks = new ArrayList<>();
//
//                    tasks.add(new DepthTask(chunk, from, mid));
//                    tasks.add(new DepthTask(chunk, mid, to));
//                    invokeAll(tasks);
//
////                    DepthTask left = new DepthTask(chunk, from, mid);
////                    DepthTask right = new DepthTask(chunk, mid, to);
////
////                    left.fork();
////                    right.compute();
////                    left.join();
//
//                    return true;
//                }
//            }
//        }
//
////        pool.invoke(new DepthTask(chunk, 0, nodes.size()));
//
//        if (verbose) {
//            System.out.println("Done with depth");
//        }

//        boolean retVal = freeDegree(nodes, adjacencies) > depth;
//
//        if (!retVal) {
////            pool.shutdown();
//        }
//
//        return retVal;
    }

    private List<Node> possibleParents(Node x, List<Node> adjx,
                                       IKnowledge knowledge) {
        List<Node> possibleParents = new LinkedList<>();
        String _x = x.getName();

        for (Node z : adjx) {
            String _z = z.getName();

            if (possibleParentOf(_z, _x, knowledge)) {
                possibleParents.add(z);
            }
        }

        return possibleParents;
    }

    private boolean possibleParentOf(String z, String x, IKnowledge knowledge) {
        return !knowledge.isForbidden(z, x) && !knowledge.isRequired(x, z);
    }

    public int getNumIndependenceTests() {
        return numIndependenceTests;
    }

    @Override
    public void setTrueGraph(Graph trueGraph) {

    }

    @Override
    public List<Node> getNodes() {
        return null;
    }

    @Override
    public List<Triple> getAmbiguousTriples(Node node) {
        return null;
    }

    public SepsetMap getSepsets() {
        return sepsets;
    }

    public void setInitialGraph(Graph initialGraph) {
        this.initialGraph = initialGraph;
    }

    /**
     * The logger, by default the empty logger.
     */
    public TetradLogger getLogger() {
        return logger;
    }

    public void setLogger(TetradLogger logger) {
        this.logger = logger;
    }

    public boolean isVerbose() {
        return verbose;
    }

    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }

    @Override
    public int getNumFalseDependenceJudgments() {
        return 0;
    }

    @Override
    public int getNumDependenceJudgments() {
        return 0;
    }

    public void setOut(PrintStream out) {
        if (out == null) throw new NullPointerException();
        this.out = out;
    }

    public PrintStream getOut() {
        return out;
    }

    /**
     * True if sepsets should be recorded. This is not necessary for all algorithms.
     */
    public boolean isRecordSepsets() {
        return recordSepsets;
    }

    public void setRecordSepsets(boolean recordSepsets) {
        this.recordSepsets = recordSepsets;
    }

    private int choose(int N, int K) {
        if (K > N) {
            return 0;
        } else if (K == N) {
            return 1;
        }
        int val = 1;
        for (int k = 0; k < K; k++) {
            val *= (N-k)/(k+1);
        }
        return val;
    }

}


