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

import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.GraphUtils;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.graph.Triple;
import edu.cmu.tetrad.util.ChoiceGenerator;
import edu.cmu.tetrad.util.ForkJoinPoolInstance;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

/**
 * Created by josephramsey on 3/24/15.
 */
public class SepsetsConservativeConcurrent implements SepsetProducer {
    private final Graph graph;
    private final IndependenceTest independenceTest;
    private final SepsetMap extraSepsets;
    private int depth = 3;
    private boolean verbose = true;
    private ConcurrentMap<Triple, List<Boolean>> tripleMap;
    private int parallelism;
    private double chunk = 5;
    private double factor = 0.8;
    private ForkJoinPool pool;
//    private ForkJoinPool pool = ForkJoinPoolInstance.getInstance().getPool();

    public SepsetsConservativeConcurrent(Graph graph, IndependenceTest independenceTest, SepsetMap extraSepsets, int depth) {
        this.graph = graph;
        this.independenceTest = independenceTest;
        this.extraSepsets = extraSepsets;
        this.depth = depth;
        this.parallelism = Runtime.getRuntime().availableProcessors();
        this.tripleMap = new ConcurrentHashMap<>();
        this.pool = new ForkJoinPool();
        System.out.println("Begin concurrent fillSepsetsListsMap");
        fillSepsetsListsMap(this.graph, this.verbose, this.tripleMap);
        System.out.println("End of concurrent fillSepsetsListsMap");
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
        Boolean ret = tripleMap.get(new Triple(i, j, k)).get(0);
        if (ret == null) {
            System.out.println("unknown triple being requested");
            List<List<List<Node>>> sepsetsLists = getSepsetsLists(i, j, k, independenceTest, depth, true);
            List<Boolean> temp = new ArrayList<>();
            temp.add(sepsetsLists.get(0).isEmpty());
            temp.add(sepsetsLists.get(1).isEmpty());
            tripleMap.put(new Triple(i, j, k), temp);
            ret = temp.get(0);
            System.out.println("unknown triple added");
        }
        return ret;
    }

    public boolean isNoncollider(Node i, Node j, Node k) {
//        List<List<List<Node>>> ret = sepsetsListsMap.get(new Triple(i, j, k));
        Boolean ret = tripleMap.get(new Triple(i, j, k)).get(1);
        if (ret == null) {
            System.out.println("unknown triple being requested");
            List<List<List<Node>>> sepsetsLists = getSepsetsLists(i, j, k, independenceTest, depth, true);
            List<Boolean> temp = new ArrayList<>();
            temp.add(sepsetsLists.get(0).isEmpty());
            temp.add(sepsetsLists.get(1).isEmpty());
            tripleMap.put(new Triple(i, j, k), temp);
            ret = temp.get(1);
//            ret = getSepsetsLists(i, j, k, independenceTest, depth, true);
//            sepsetsListsMap.put(new Triple(i, j, k), ret);
            System.out.println("unknown triple added");
        }
        return ret;
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


    private void fillSepsetsListsMap(final Graph graph, boolean verbose, final Map<Triple, List<Boolean>> tripleMap) {

        final HashMap<Integer,Integer> powerSetSizes = new HashMap<>();
        final List<Node> nodes = graph.getNodes();
        int sum = 0;
        int psSum;
        for(int i = 0; i < nodes.size();i++)
        {
            psSum = 0;
            List<Node> adjNodes = graph.getAdjacentNodes(nodes.get(i));
            for (int j = 0; j < adjNodes.size(); j++)
            {
                psSum +=  Math.pow(2, graph.getAdjacentNodes(adjNodes.get(j)).size()-1);
            }
            powerSetSizes.put(i, psSum);
            sum += psSum;
        }
        chunk = sum/(parallelism) * factor;

        class colliderTask extends RecursiveTask<Boolean> {
            private double chunk;
            private int from;
            private int to;
            public colliderTask(double chunk,int from, int to)
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
                        final Node b = nodes.get(i);
                        final List<Node> adjacent = graph.getAdjacentNodes(b);
                        //only compare to nodes less than it in index
                        if(adjacent.size()<2)
                            continue;
                        ChoiceGenerator cg = new ChoiceGenerator(adjacent.size(),2);
                        int [] combination;
                        while((combination = cg.next()) != null)
                        {
                            final  Node a = adjacent.get(combination[0]);
                            final  Node c = adjacent.get(combination[1]);
                            if(graph.isAdjacentTo(a,c))
                                continue;

                            List<List<List<Node>>> ret = getSepsetsLists(a, b, c, independenceTest, depth, verbose);

                            List<Boolean> temp = new ArrayList<>();
                            temp.add(ret.get(0).isEmpty());
                            temp.add(ret.get(1).isEmpty());
                            tripleMap.put(new Triple(a, b, c), temp);

//                            sepsetsListsMap.put(new Triple(a,b,c), ret);
                        }
                    }
                    System.out.println(Thread.currentThread().getName() + ": Finished");
                    return true;
                }
                else
                {
                    List<colliderTask> tasks = new ArrayList<>();
                    final int mid = (to+from)/2;
                    colliderTask t1 = new colliderTask(chunk,from,mid);
                    tasks.add(t1);
                    colliderTask t2 = new colliderTask(chunk,mid,to);
                    tasks.add(t2);
                    invokeAll(tasks);
                    return true;
                }

            }
        }
        pool.invoke(new colliderTask(chunk,0,nodes.size()));
        pool.shutdown();
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

