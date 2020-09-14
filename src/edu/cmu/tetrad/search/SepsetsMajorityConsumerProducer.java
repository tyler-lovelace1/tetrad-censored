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

import edu.cmu.tetrad.data.ContinuousVariable;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.GraphUtils;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.graph.Triple;
import edu.cmu.tetrad.util.ChoiceGenerator;
import org.apache.commons.lang3.RandomStringUtils;

import java.util.*;
import java.util.concurrent.*;

/**
 * Created by josephramsey on 3/24/15.
 */
public class SepsetsMajorityConsumerProducer implements SepsetProducer {
    private final Graph graph;
    private final IndependenceTest independenceTest;
    private final SepsetMap extraSepsets;
    private int depth = 3;
    private boolean verbose = true;
    private ConcurrentMap<Triple, List<Integer>> tripleMap;
    private ConcurrentMap<Set<Node>, List<Node>> maxPSepsetMap;
    private int parallelism;
    private ExecutorService executorService;
//    private double chunk = 5;
//    private double factor = 1.25;
//    private ForkJoinPool pool;
//    private ForkJoinPool pool = ForkJoinPoolInstance.getInstance().getPool();

    public SepsetsMajorityConsumerProducer(Graph graph, IndependenceTest independenceTest, SepsetMap extraSepsets, int depth) {
        this.graph = graph;
        this.independenceTest = independenceTest;
        this.extraSepsets = extraSepsets;
        this.depth = depth;
        this.parallelism = Runtime.getRuntime().availableProcessors();
        this.tripleMap = new ConcurrentHashMap<>();
        this.executorService = Executors.newFixedThreadPool(parallelism+1);
        System.out.println("Begin concurrent fillSepsetsCountsMap");
        fillSepsetsCountsMap(this.graph, this.verbose, this.tripleMap);
        System.out.println("End of concurrent fillSepsetsCountsMap");
    }

    /**
     * Pick out the sepset from among adj(i) or adj(k) with the highest p value.
     */
    public List<Node> getSepset(Node i, Node k) {
        double _p = 0.0;
        List<Node> _v = null;

        if (extraSepsets != null) {
            final List<Node> possibleDsep = extraSepsets.get(i, k);
            if (possibleDsep != null) {
                independenceTest.isIndependent(i, k, possibleDsep);
                _p = independenceTest.getPValue();
                _v = possibleDsep;
            }
        }

        List<Node> adji = graph.getAdjacentNodes(i);
        List<Node> adjk = graph.getAdjacentNodes(k);
        adji.remove(k);
        adjk.remove(i);

        for (int d = 0; d <= Math.min((depth == -1 ? 1000 : depth), Math.max(adji.size(), adjk.size())); d++) {
            if (d <= adji.size()) {
                ChoiceGenerator gen = new ChoiceGenerator(adji.size(), d);
                int[] choice;

                while ((choice = gen.next()) != null) {
                    List<Node> v = GraphUtils.asList(choice, adji);

                    if (getIndependenceTest().isIndependent(i, k, v)) {
                        double pValue = getIndependenceTest().getPValue();
                        if (pValue > _p) {
                            _p = pValue;
                            _v = v;
                        }
                    }
                }
            }

            if (d <= adjk.size()) {
                ChoiceGenerator gen = new ChoiceGenerator(adjk.size(), d);
                int[] choice;

                while ((choice = gen.next()) != null) {
                    List<Node> v = GraphUtils.asList(choice, adjk);
                    if (getIndependenceTest().isIndependent(i, k, v)) {
                        double pValue = getIndependenceTest().getPValue();
                        if (pValue > _p) {
                            _p = pValue;
                            _v = v;
                        }
                    }
                }
            }
        }

        return _v;
    }

    public boolean isCollider(Node i, Node j, Node k) {
        if (j.getName()=="survival"){
            System.out.println("checking for collider centered on survival");
        }
        List<Integer> ret = tripleMap.get(new Triple(i, j, k));
        if (j.getName()=="survival"){
            System.out.println("number of sepsets with survival:\t" + ret.get(0));
            System.out.println("number of sepsets without survival:\t" + ret.get(1));
        }
        if (ret == null) {
            System.out.println("unknown triple being requested");
            List<List<List<Node>>> sepsetsLists = getSepsetsLists(i, j, k, independenceTest, depth, true);
            List<Integer> temp = new ArrayList<>();
            temp.add(sepsetsLists.get(0).size());
            temp.add(sepsetsLists.get(1).size());
            tripleMap.put(new Triple(i, j, k), temp);
            ret = temp;
            System.out.println("unknown triple added");
        }
        return ret.get(0) < ret.get(1);
    }

    public boolean isNoncollider(Node i, Node j, Node k) {
//        List<List<List<Node>>> ret = sepsetsListsMap.get(new Triple(i, j, k));
        List<Integer> ret = tripleMap.get(new Triple(i, j, k));
        if (ret == null) {
            System.out.println("unknown triple being requested");
            List<List<List<Node>>> sepsetsLists = getSepsetsLists(i, j, k, independenceTest, depth, true);
            List<Integer> temp = new ArrayList<>();
            temp.add(sepsetsLists.get(0).size());
            temp.add(sepsetsLists.get(1).size());
            tripleMap.put(new Triple(i, j, k), temp);
            ret = temp;
//            ret = getSepsetsLists(i, j, k, independenceTest, depth, true);
//            sepsetsListsMap.put(new Triple(i, j, k), ret);
            System.out.println("unknown triple added");
        }
        return ret.get(0) >= ret.get(1);
    }

    // The published version.
    public List<List<List<Node>>> getSepsetsLists(Node x, Node y, Node z,
                                                  IndependenceTest test, int depth,
                                                  boolean verbose) {
        List<List<Node>> sepsetsContainingY = new ArrayList<>();
        List<List<Node>> sepsetsNotContainingY = new ArrayList<>();

        List<Node> _nodes = graph.getAdjacentNodes(x);
        _nodes.remove(z);

        int _depth = depth;
        if (_depth == -1) {
            _depth = 1000;
        }

        _depth = Math.min(_depth, _nodes.size());

        for (int d = 0; d <= _depth; d++) {
            ChoiceGenerator cg = new ChoiceGenerator(_nodes.size(), d);
            int[] choice;

            while ((choice = cg.next()) != null) {
                List<Node> cond = GraphUtils.asList(choice, _nodes);

                if (test.isIndependent(x, z, cond)) {
                    if (verbose) {
                        System.out.println("Indep: " + x + " _||_ " + z + " | " + cond);
                    }

                    if (cond.contains(y)) {
                        sepsetsContainingY.add(cond);
                    } else {
                        sepsetsNotContainingY.add(cond);
                    }
                }
            }
        }

        _nodes = graph.getAdjacentNodes(z);
        _nodes.remove(x);

        _depth = depth;
        if (_depth == -1) {
            _depth = 1000;
        }
        _depth = Math.min(_depth, _nodes.size());

        for (int d = 0; d <= _depth; d++) {
            ChoiceGenerator cg = new ChoiceGenerator(_nodes.size(), d);
            int[] choice;

            while ((choice = cg.next()) != null) {
                List<Node> cond = GraphUtils.asList(choice, _nodes);

                if (test.isIndependent(x, z, cond)) {
                    if (cond.contains(y)) {
                        sepsetsContainingY.add(cond);
                    } else {
                        sepsetsNotContainingY.add(cond);
                    }
                }
            }
        }

        List<List<List<Node>>> ret = new ArrayList<>();
        ret.add(sepsetsContainingY);
        ret.add(sepsetsNotContainingY);

        return ret;
    }

    class ColliderTask {
        public Triple triple;
        public Node x;
        public Node y;
        public List<Node> z;

        public ColliderTask(Triple triple, Node x, Node y, List<Node> z) {
            this.triple = triple;
            this.x = x;
            this.y = y;
            this.z = z;
        }

        @Override
        public boolean equals(Object o) {
            if (o == null) {
                return false;
            } else if (!(o instanceof ColliderTask)) {
                return false;
            } else if ( ((ColliderTask) o).triple != this.triple) {
                return false;
            } else if ( ((ColliderTask) o).x != this.x) {
                return false;
            } else if ( ((ColliderTask) o).y != this.y) {
                return false;
            } else if ( ((ColliderTask) o).z != this.z) {
                return false;
            }
            return true;
        }

        @Override
        public String toString() {
            return String.format(triple + " | " + z);
        }
    }

    private class Broker {
        public ArrayBlockingQueue<ColliderTask> queue = new ArrayBlockingQueue<>(1000000);

        public void put(ColliderTask task) throws InterruptedException {
            queue.put(task);
        }

        public ColliderTask get() throws InterruptedException {
            return queue.poll(1, TimeUnit.SECONDS);
        }
    }

    private class SepsetsCountsProducer implements Runnable {
        private Broker broker;
        private Graph graph;
        private Map<Triple, List<Integer>> tripleMap;
        private ColliderTask poisonPill;

        public SepsetsCountsProducer(Broker broker, final Graph graph, final Map<Triple, List<Integer>> tripleMap,
                                     final ColliderTask poisonPill) {
            this.broker = broker;
            this.graph = graph;
            this.tripleMap = tripleMap;
            this.poisonPill = poisonPill;
        }

        @Override
        public void run() {
            System.out.println("\t" + Thread.currentThread().getName() + ": SepsetsCountsProducer Start");
            final List<Node> nodes = graph.getNodes();
            try {
                for (int i = 0; i < nodes.size(); i ++) {
                    final Node b = nodes.get(i);
                    final List<Node> adjacent = graph.getAdjacentNodes(b);
                    //only compare to nodes less than it in index
                    if (adjacent.size() < 2)
                        continue;
                    ChoiceGenerator cg = new ChoiceGenerator(adjacent.size(), 2);
                    int[] combination;
                    while ((combination = cg.next()) != null) {
                        final Node a = adjacent.get(combination[0]);
                        final Node c = adjacent.get(combination[1]);
                        if (graph.isAdjacentTo(a, c))
                            continue;

                        List<Integer> sepsetCounts = new ArrayList<>();
                        sepsetCounts.add(0);
                        sepsetCounts.add(0);
                        Triple curTriple = new Triple(a, b, c);
                        tripleMap.put(curTriple, sepsetCounts);

                        List<Node> _nodes = graph.getAdjacentNodes(a);
                        _nodes.remove(c);

                        int _depth = depth;
                        if (_depth == -1) {
                            _depth = 1000;
                        }

                        _depth = Math.min(_depth, _nodes.size());

                        for (int d = 0; d <= _depth; d++) {
                            ChoiceGenerator cg2 = new ChoiceGenerator(_nodes.size(), d);
                            int[] choice;

                            while ((choice = cg2.next()) != null) {
                                List<Node> cond = GraphUtils.asList(choice, _nodes);

                                broker.put(new ColliderTask(curTriple, a, c, cond));

                            }
                        }

                        _nodes = graph.getAdjacentNodes(c);
                        _nodes.remove(a);

                        _depth = depth;
                        if (_depth == -1) {
                            _depth = 1000;
                        }
                        _depth = Math.min(_depth, _nodes.size());

                        for (int d = 0; d <= _depth; d++) {
                            ChoiceGenerator cg2 = new ChoiceGenerator(_nodes.size(), d);
                            int[] choice;

                            while ((choice = cg2.next()) != null) {
                                List<Node> cond = GraphUtils.asList(choice, _nodes);

                                broker.put(new ColliderTask(curTriple, a, c, cond));

                            }
                        }

                    }
                }
                broker.put(poisonPill);
                broker.put(poisonPill);
                System.out.println("\t" + Thread.currentThread().getName() + ": SepsetsCountsProducer Finish");
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    private class SepsetsCountsConsumer implements Runnable {
        private Broker broker;
        private boolean verbose;
        private Map<Triple, List<Integer>> tripleMap;
        private ColliderTask poisonPill;

        public SepsetsCountsConsumer(Broker broker, boolean verbose, final Map<Triple, List<Integer>> tripleMap,
                                     final ColliderTask poisonPill) {
            this.broker = broker;
            this.verbose = verbose;
            this.tripleMap = tripleMap;
            this.poisonPill = poisonPill;
        }

        @Override
        public void run() {
            System.out.println("\t" + Thread.currentThread().getName() + ": SepsetsCountsConsumer Start");
            try {
                ColliderTask task = broker.get();

                while (task != poisonPill) {

                    if (task == null) {
                        task = broker.get();
                        continue;
                    }

                    if (independenceTest.isIndependent(task.x, task.y, task.z)) {
                        if (verbose) {
                            System.out.println("Indep: " + task.x + " _||_ " + task.y + " | " + task.z);
                        }

                        if (task.z.contains(task.triple.getY())) {
                            tripleMap.get(task.triple).set(0, tripleMap.get(task.triple).get(0)+1);
                        } else {
                            tripleMap.get(task.triple).set(1, tripleMap.get(task.triple).get(1)+1);
                        }
                    }

                    task = broker.get();
                }
                broker.put(poisonPill);
                broker.put(poisonPill);
                System.out.println("\t" + Thread.currentThread().getName() + ": SepsetsCountsConsumer Finish");
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }


    private void fillSepsetsCountsMap(final Graph graph, boolean verbose, final Map<Triple, List<Integer>> tripleMap) {

        Broker broker = new Broker();
        List<Future> status = new ArrayList<>();

        Node x = new ContinuousVariable(RandomStringUtils.randomAlphabetic(15));
        Node y = new ContinuousVariable(RandomStringUtils.randomAlphabetic(15));
        Node z = new ContinuousVariable(RandomStringUtils.randomAlphabetic(15));

        ColliderTask poisonPill = new ColliderTask(new Triple(x,y,z), x, z, new ArrayList<>());

        System.out.println("PoisonPill: " + poisonPill);

        try {
            status.add(executorService.submit(new SepsetsCountsProducer(broker, graph, tripleMap, poisonPill)));
            for (int i = 0; i < parallelism; i++) {
                status.add(executorService.submit(new SepsetsCountsConsumer(broker, verbose, tripleMap, poisonPill)));
            }

            for (int i = 0; i < parallelism+1; i++) {
                status.get(i).get();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        executorService.shutdown();

//        final HashMap<Integer,Integer> powerSetSizes = new HashMap<>();
//        final List<Node> nodes = graph.getNodes();
//        int sum = 0;
//        int psSum;
//        for(int i = 0; i < nodes.size();i++)
//        {
//            psSum = 0;
//            List<Node> adjNodes = graph.getAdjacentNodes(nodes.get(i));
//            for (int j = 0; j < adjNodes.size(); j++)
//            {
//                psSum +=  Math.pow(2, graph.getAdjacentNodes(adjNodes.get(j)).size()-1);
//            }
//            powerSetSizes.put(i, psSum);
//            sum += psSum;
//        }
//        chunk = sum/(parallelism) * factor;
//
//        class colliderTask extends RecursiveTask<Boolean> {
//            private double chunk;
//            private int from;
//            private int to;
//            public colliderTask(double chunk,int from, int to)
//            {
//                this.chunk = chunk;
//                this.from = from;
//                this.to = to;
//            }
//            protected Boolean compute()
//            {
//                int numTests = 0;
//                for(int j = from; j < to; j++)
//                {
//                    numTests += powerSetSizes.get(j);
//                }
//                if(to-from <= 2 || numTests <= chunk)
//                {
//                    System.out.println(Thread.currentThread().getName() + ": " + numTests + " / " + chunk);
//                    for(int i = from; i < to; i ++)
//                    {
//                        final Node b = nodes.get(i);
//                        final List<Node> adjacent = graph.getAdjacentNodes(b);
//                        //only compare to nodes less than it in index
//                        if(adjacent.size()<2)
//                            continue;
//                        ChoiceGenerator cg = new ChoiceGenerator(adjacent.size(),2);
//                        int [] combination;
//                        while((combination = cg.next()) != null)
//                        {
//                            final  Node a = adjacent.get(combination[0]);
//                            final  Node c = adjacent.get(combination[1]);
//                            if(graph.isAdjacentTo(a,c))
//                                continue;
//
//                            List<List<List<Node>>> ret = getSepsetsLists(a, b, c, independenceTest, depth, verbose);
//
//                            List<Integer> temp = new ArrayList<>();
//                            temp.add(ret.get(0).size());
//                            temp.add(ret.get(1).size());
//                            tripleMap.put(new Triple(a, b, c), temp);
//
//                            if (temp.get(0) == temp.get(1)) {
//                                System.out.println("Ambiguous Triple: " + new Triple(a, b, c) + ": " + temp.get(0) + ", " + temp.get(1));
//                            }
//
////                            sepsetsListsMap.put(new Triple(a,b,c), ret);
//                        }
//                    }
//                    System.out.println(Thread.currentThread().getName() + ": Finished");
//                    return true;
//                }
//                else
//                {
//                    List<colliderTask> tasks = new ArrayList<>();
//                    final int mid = (to+from)/2;
//                    colliderTask t1 = new colliderTask(chunk,from,mid);
//                    tasks.add(t1);
//                    colliderTask t2 = new colliderTask(chunk,mid,to);
//                    tasks.add(t2);
//                    invokeAll(tasks);
//                    return true;
//                }
//
//            }
//        }
//        pool.invoke(new colliderTask(chunk,0,nodes.size()));
    }


    @Override
    public boolean isIndependent(Node a, Node b, List<Node> c) {
        return independenceTest.isIndependent(a, b, c);
    }

    @Override
    public double getPValue() {
        return independenceTest.getPValue();
    }

    @Override
    public double getScore() {
        return -(independenceTest.getPValue() - independenceTest.getAlpha());
    }

    @Override
    public List<Node> getVariables() {
        return independenceTest.getVariables();
    }

    @Override
    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }

    public IndependenceTest getIndependenceTest() {
        return independenceTest;
    }
}

