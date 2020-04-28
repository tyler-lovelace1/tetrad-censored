package edu.pitt.csb.stability;

import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import edu.cmu.tetrad.data.ColtDataSet;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.Fci;
import edu.cmu.tetrad.search.FciMaxP;
import edu.cmu.tetrad.search.IndependenceTest;
import edu.cmu.tetrad.util.ForkJoinPoolInstance;
import edu.pitt.csb.Priors.runPriors;
import edu.pitt.csb.latents.LatentPrediction;
import edu.pitt.csb.mgm.IndTestMultinomialAJ;
import edu.pitt.csb.mgm.MGM;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

/**
 * Created by vinee_000 on 7/12/2018.
 * Class to run Complimentary Pairs Stability Selection with causal modeling algorithms
 *
 */
public class CPSS {
    private int B = 50; //Number of subsamples
    private double theta; //q/p is the suggestion
    private int p;//total number of variables
    private int q; //average number of selected variables
    private double tao; //threshold that will be calculated from q, theta, and error rate
    private double bound; //error rate as specified in the paper
    private int boundIndex;
    private DataSet data;
    private double [] lambda; //Array of lambda values for MGM
    private int [][] subs;
//TODO need to modify this in a similar way as Bootstrap to get compatibility with runAlgorithms

    public CPSS(DataSet data, double [] lambda, double bound)
    {
        this.data = data;
        this.lambda = lambda;
        this.bound = bound;
        this.boundIndex = getBoundIndex(bound);
        this.p = data.getNumColumns()*(data.getNumColumns()-1)/2;
    }

    public CPSS(DataSet data, double [] lambda, double bound, int [][]subs)
    {
        this.data = data;
        this.lambda = lambda;
        this.bound = bound;
        this.boundIndex = getBoundIndex(bound);
        this.p = data.getNumColumns()*(data.getNumColumns()-1)/2;
        this.subs = subs;
    }

    private static int getBoundIndex(double bound)
    {
        int boundIndex = -1;
        if(bound<=0.001)
            boundIndex=0;
        else if(bound<=0.01)
            boundIndex=1;
        else if(bound <=0.05)
            boundIndex=2;
        else
            boundIndex = 3;
        return boundIndex;
    }
    public static int[][] createSubs(DataSet data, int B)
    {
        ArrayList<Integer>tempInds  = new ArrayList<Integer>();
        for(int i = 0; i < data.getNumRows();i++)
            tempInds.add(i);
        int [][] subs = new int[B*2][];
        for(int i = 0; i < B;i++)
        {
            Collections.shuffle(tempInds);
            int[] d1 = new int[tempInds.size() / 2];
            for (int j = 0; j < d1.length; j++) {
                d1[j] = tempInds.get(j);
            }
            int size2 = tempInds.size() / 2;
            if (tempInds.size() % 2 == 1)
                size2 = tempInds.size() / 2 + 1;
            int[] d2 = new int[size2];
            for (int j = d1.length; j < tempInds.size(); j++) {
                d2[j - d1.length] = tempInds.get(j);
            }

            subs[2*i] = d1;
            subs[2*i+1] = d2;
            if(runPriors.checkForVariance(data.subsetRows(d1),data)!=-1 || runPriors.checkForVariance(data.subsetRows(d2),data)!=-1)
                --i;
        }
        return subs;
    }
    private int[][] createSubs(DataSet data)
    {
        return createSubs(data,B);
    }
    public ArrayList<LatentPrediction.Pair> runCPSSLatent(final String algName, double alpha)
    {
        final double tempAlpha = alpha;
        final double [] tempLambda = lambda;
        if(subs==null)
            subs = createSubs(data);
        final ArrayList<Graph> graphs = new ArrayList<Graph>();
        final DoubleMatrix2D edgeCounts = new SparseDoubleMatrix2D(data.getNumColumns(),data.getNumColumns());
        final int [] totalVars = new int[B];
        System.out.print("Computing " + B*2 + " Graphs in parallel... using " + algName);



        final ForkJoinPool pool = ForkJoinPoolInstance.getInstance().getPool();

        class StabilityAction extends RecursiveAction {
            private int chunk;
            private int from;
            private int to;

            public StabilityAction(int chunk, int from, int to) {
                this.chunk = chunk;
                this.from = from;
                this.to = to;
            }

            //could avoid using synchronized if we keep track of array of mats and add at end, but that needs lots of
            //memory

            private synchronized ArrayList<Integer> createTemp(ArrayList<Integer> inds) {
                ArrayList<Integer> temp = new ArrayList<Integer>();
                temp.addAll(inds);
                return temp;
            }

            private synchronized void addToGraphs(Graph g){graphs.add(g);}
            private synchronized void addToMat(Graph g, DataSet data, DoubleMatrix2D edgeCounts)
            {
                for(Edge e:g.getEdges())
                {
                    int x = data.getColumn(data.getVariable(e.getNode1().getName()));
                    int y = data.getColumn(data.getVariable(e.getNode2().getName()));
                    try {
                        edgeCounts.set(x, y, edgeCounts.get(x, y) + 1);
                        edgeCounts.set(y, x, edgeCounts.get(y, x) + 1);
                    }
                    catch(Exception e2)
                    {
                        System.out.println(e + "\t" + data + "\t" + g);
                        System.out.println(x + "\t" + y);
                        System.out.println(data.getVariable(e.getNode1().getName()));
                        System.out.println(data.getVariable(e.getNode2().getName()));
                        e2.printStackTrace();
                        System.exit(-1);
                    }
                }
            }

            private synchronized int getLatentCount(Graph g1)
            {
                int result = 0;
                for(Edge e:g1.getEdges())
                {
                    if(e.getEndpoint1()== Endpoint.ARROW && e.getEndpoint2()==Endpoint.ARROW)
                    {
                        result++;
                    }
                }
                return result;
            }
            private synchronized DataSet createData(){return new ColtDataSet(data.getNumColumns(),data.getVariables());}

            private synchronized DataSet subset(int [] x){return data.subsetRows(x);}
            @Override
            protected void compute(){
                if (to - from <= chunk) {
                    for (int s = from; s < to; s++) {
                            DataSet data1 = subset(subs[2*s]);
                            DataSet data2 = subset(subs[2*s+1]);
                        Graph g1 = new EdgeListGraph(data.getVariables());
                        Graph g2 = new EdgeListGraph(data.getVariables());
                        if(algName.equals("FCI") || algName.equals("MGM-FCI"))
                        {
                            IndependenceTest ind = new IndTestMultinomialAJ(data1,tempAlpha);
                            Fci f = new Fci(ind);
                            if(algName.equals("MGM-FCI"))
                            {
                                MGM m = new MGM(data1,tempLambda);
                                m.learnEdges(1000);
                                f.setInitialGraph(m.graphFromMGM());
                            }
                            g1 = f.search();
                            ind = new IndTestMultinomialAJ(data2,tempAlpha);
                            f = new Fci(ind);
                            if(algName.equals("MGM-FCI"))
                            {
                                MGM m = new MGM(data2,tempLambda);
                                m.learnEdges(1000);
                                f.setInitialGraph(m.graphFromMGM());
                            }
                            g2 = f.search();
                        }
                        else
                        {
                            IndependenceTest ind = new IndTestMultinomialAJ(data1,tempAlpha);
                            FciMaxP f = new FciMaxP(ind);
                            MGM m = new MGM(data1,tempLambda);
                            m.learnEdges(1000);
                            f.setInitialGraph(m.graphFromMGM());
                            g1 = f.search();
                            ind = new IndTestMultinomialAJ(data2,tempAlpha);
                            f = new FciMaxP(ind);
                            m = new MGM(data2,tempLambda);
                            m.learnEdges(1000);
                            f.setInitialGraph(m.graphFromMGM());
                            g2 = f.search();
                        }
                        totalVars[s]+=getLatentCount(g1);
                        //addToMat(g,data,edgeCounts);
                        addToGraphs(g1);
                        addToGraphs(g2);
                        totalVars[s]+=getLatentCount(g2);
                        //addToMat(g,data,edgeCounts);
                    }

                    return;
                } else {
                    List<StabilityAction> tasks = new ArrayList<>();

                    final int mid = (to + from) / 2;

                    tasks.add(new StabilityAction(chunk, from, mid));
                    tasks.add(new StabilityAction(chunk, mid, to));

                    invokeAll(tasks);

                    return;
                }
            }

        }

        final int chunk = 2;

        pool.invoke(new StabilityAction(chunk, 0,B));
        int sum = 0;
        for(int i = 0; i < totalVars.length;i++)
        {
            sum+=totalVars[i];
        }
        for(int i = 0 ; i< graphs.size();i++)
        {
            Graph curr = graphs.get(i);
            for(Edge e: curr.getEdges())
            {
                if(e.getEndpoint1()==Endpoint.ARROW && e.getEndpoint2()==Endpoint.ARROW) {
                    int x = data.getColumn(data.getVariable(e.getNode1().getName()));
                    int y = data.getColumn(data.getVariable(e.getNode2().getName()));
                    edgeCounts.set(x, y, edgeCounts.get(x, y) + 1);
                    edgeCounts.set(y, x, edgeCounts.get(y, x) + 1);
                }
            }
        }
        System.out.println("Done");
        double avgVars = sum/(double)(B*2);
        double p = data.getNumColumns()*(data.getNumColumns()-1)/2;
        double theta = avgVars/p;
        System.out.println("Avg # Edges: " + avgVars + ", Total # of Vars: " + totalVars);
        System.out.println("Theta is: " + theta);
        tao = computeTao();
        System.out.println("Tao is: " + tao);
        ArrayList<LatentPrediction.Pair> finalResult = new ArrayList<LatentPrediction.Pair>();
        for(int i = 0; i < edgeCounts.rows();i++)
        {
            for(int j = i+1; j < edgeCounts.columns();j++)
            {
                if(edgeCounts.get(i,j)/(B*2)>=tao)
                {
                    finalResult.add(new LatentPrediction.Pair(data.getVariable(i),data.getVariable(j),edgeCounts.get(i,j)/(B*2)));
                }
            }
        }
        //Partition the data B times and run MGM on each partition to get a count of each edge's appearence
        //Based on the average number of selected variables per run, compute theta
        //use theta, q, and the bound to get a value for tao
        //only keep the edges with probability greater than tao
        //Return the graph with these edges
        return finalResult;
    }

    public Graph runCPSSPar()
    {
        final ArrayList<Integer>inds  = new ArrayList<Integer>();
        final ArrayList<Graph> graphs = new ArrayList<Graph>();
        for(int i = 0; i < data.getNumRows();i++)
        {
            inds.add(i);
        }
        final DoubleMatrix2D edgeCounts = new SparseDoubleMatrix2D(data.getNumColumns(),data.getNumColumns());
        final int [] totalVars = new int[B];
        System.out.print("Computing " + B*2 + " MGM Graphs in parallel...");



        final ForkJoinPool pool = ForkJoinPoolInstance.getInstance().getPool();

        class StabilityAction extends RecursiveAction {
            private int chunk;
            private int from;
            private int to;

            public StabilityAction(int chunk, int from, int to) {
                this.chunk = chunk;
                this.from = from;
                this.to = to;
            }

            //could avoid using synchronized if we keep track of array of mats and add at end, but that needs lots of
            //memory

            private synchronized ArrayList<Integer> createTemp(ArrayList<Integer> inds) {
                ArrayList<Integer> temp = new ArrayList<Integer>();
                temp.addAll(inds);
                return temp;
            }

            private synchronized void addToGraphs(Graph g){graphs.add(g);}
            private synchronized void addToMat(Graph g, DataSet data, DoubleMatrix2D edgeCounts)
            {
                for(Edge e:g.getEdges())
                {
                    int x = data.getColumn(data.getVariable(e.getNode1().getName()));
                    int y = data.getColumn(data.getVariable(e.getNode2().getName()));
                    try {
                        edgeCounts.set(x, y, edgeCounts.get(x, y) + 1);
                        edgeCounts.set(y, x, edgeCounts.get(y, x) + 1);
                    }
                    catch(Exception e2)
                    {
                        System.out.println(e + "\t" + data + "\t" + g);
                        System.out.println(x + "\t" + y);
                        System.out.println(data.getVariable(e.getNode1().getName()));
                        System.out.println(data.getVariable(e.getNode2().getName()));
                        e2.printStackTrace();
                        System.exit(-1);
                    }
                }
            }

            private synchronized DataSet createData(){return new ColtDataSet(data.getNumColumns(),data.getVariables());}

            private synchronized DataSet subset(int [] x){return data.subsetRows(x);}
            @Override
            protected void compute(){
                if (to - from <= chunk) {
                    for (int s = from; s < to; s++) {
                        //System.out.println(s);
                        ArrayList<Integer> tempInds = createTemp(inds);
                        DataSet data1 = createData();
                        DataSet data2 = createData();
                        boolean done = false;
                        while(!done) {
                            Collections.shuffle(tempInds);
                            int[] d1 = new int[tempInds.size() / 2];
                            for (int j = 0; j < d1.length; j++) {
                                d1[j] = tempInds.get(j);
                            }
                            int size2 = tempInds.size() / 2;
                            if (tempInds.size() % 2 == 1)
                                size2 = tempInds.size() / 2 + 1;
                            int[] d2 = new int[size2];
                            for (int j = d1.length; j < tempInds.size(); j++) {
                                d2[j - d1.length] = tempInds.get(j);
                            }

                            data1 = subset(d1);
                            data2 = subset(d2);
                            done = true;
                            if(runPriors.checkForVariance(data1,data)!=-1 || runPriors.checkForVariance(data2,data)!=-1)
                                done = false;
                        }
                        MGM m = new MGM(data1,lambda);
                        m.learnEdges(1000);
                        Graph g = m.graphFromMGM();
                        totalVars[s]+=g.getNumEdges();
                        //addToMat(g,data,edgeCounts);
                        addToGraphs(g);
                        m = new MGM(data2,lambda);
                        m.learnEdges(1000);
                        g = m.graphFromMGM();
                        addToGraphs(g);
                        totalVars[s]+=g.getNumEdges();
                        //addToMat(g,data,edgeCounts);
                    }

                    return;
                } else {
                    List<StabilityAction> tasks = new ArrayList<>();

                    final int mid = (to + from) / 2;

                    tasks.add(new StabilityAction(chunk, from, mid));
                    tasks.add(new StabilityAction(chunk, mid, to));

                    invokeAll(tasks);

                    return;
                }
            }

        }

        final int chunk = 2;

        pool.invoke(new StabilityAction(chunk, 0,B));
        int sum = 0;
        for(int i = 0; i < totalVars.length;i++)
        {
            sum+=totalVars[i];
        }
        for(int i = 0 ; i< graphs.size();i++)
        {
            Graph curr = graphs.get(i);
            for(Edge e: curr.getEdges())
            {
                int x = data.getColumn(data.getVariable(e.getNode1().getName()));
                int y = data.getColumn(data.getVariable(e.getNode2().getName()));
                edgeCounts.set(x, y, edgeCounts.get(x, y) + 1);
                edgeCounts.set(y, x, edgeCounts.get(y, x) + 1);
            }
        }
        System.out.println("Done");
        double avgVars = sum/(double)(B*2);
        theta = avgVars/p;
        System.out.println("Avg # Edges: " + avgVars + ", Total # of Vars: " + totalVars);
        System.out.println("Theta is: " + theta);
        tao = computeTao();
        System.out.println("Tao is: " + tao);
        Graph finalGraph = new EdgeListGraphSingleConnections(data.getVariables());
        for(int i = 0; i < edgeCounts.rows();i++)
        {
            for(int j = i+1; j < edgeCounts.columns();j++)
            {
                if(edgeCounts.get(i,j)/(B*2)>=tao)
                {
                    finalGraph.addUndirectedEdge(data.getVariable(i),data.getVariable(j));
                }
            }
        }
        //Partition the data B times and run MGM on each partition to get a count of each edge's appearence
        //Based on the average number of selected variables per run, compute theta
        //use theta, q, and the bound to get a value for tao
        //only keep the edges with probability greater than tao
        //Return the graph with these edges
        return finalGraph;

    }
    public Graph runCPSS()
    {
        ArrayList<Integer>inds  = new ArrayList<Integer>();
        for(int i = 0; i < data.getNumRows();i++)
        {
            inds.add(i);
        }
        double [][] edgeCounts = new double[data.getNumColumns()][data.getNumColumns()];
        int totalVars = 0;
        System.out.print("Computing " + B*2 + " MGM Graphs...");
        for(int i = 0; i < B;i++)
        {
            DataSet data1 = new ColtDataSet(data.getNumColumns(),data.getVariables());
            DataSet data2 = new ColtDataSet(data.getNumColumns(),data.getVariables());
            boolean done = false;
            while(!done) {
                Collections.shuffle(inds);
                int[] d1 = new int[inds.size() / 2];
                for (int j = 0; j < d1.length; j++) {
                    d1[j] = inds.get(j);
                }
                int size2 = inds.size() / 2;
                if (inds.size() % 2 == 1)
                    size2 = inds.size() / 2 + 1;
                int[] d2 = new int[size2];
                for (int j = d1.length; j < inds.size(); j++) {
                    d2[j - d1.length] = inds.get(j);
                }

                data1 = data.subsetRows(d1);
                data2 = data.subsetRows(d2);
                done = true;
                if(runPriors.checkForVariance(data1,data)!=-1 || runPriors.checkForVariance(data2,data)!=-1)
                    done = false;
            }
            MGM m = new MGM(data1,lambda);
            m.learnEdges(1000);
            Graph g = m.graphFromMGM();
            totalVars+=g.getNumEdges();
            edgeCounts = addToMat(g,data,edgeCounts);
            m = new MGM(data2,lambda);
            m.learnEdges(1000);
            g = m.graphFromMGM();
            totalVars+=g.getNumEdges();
            edgeCounts = addToMat(g,data,edgeCounts);


        }
        System.out.println("Done");
        double avgVars = totalVars/(double)(B*2);
        theta = avgVars/p;
        System.out.println("Avg # Edges: " + avgVars + ", Total # of Vars: " + totalVars);
        System.out.println("Theta is: " + theta);
        tao = computeTao();
        System.out.println("Tao is: " + tao);
        Graph finalGraph = new EdgeListGraphSingleConnections(data.getVariables());
        for(int i = 0; i < edgeCounts.length;i++)
        {
            for(int j = i+1; j < edgeCounts.length;j++)
            {
                if(edgeCounts[i][j]/(B*2)>=tao)
                {
                    finalGraph.addUndirectedEdge(data.getVariable(i),data.getVariable(j));
                }
            }
        }
        //Partition the data B times and run MGM on each partition to get a count of each edge's appearence
        //Based on the average number of selected variables per run, compute theta
        //use theta, q, and the bound to get a value for tao
        //only keep the edges with probability greater than tao
        //Return the graph with these edges
        return finalGraph;
    }

    public static double computeTao(double theta,double bound)
    {
        final int boundIndex = getBoundIndex(bound);
        final double thetaInc = 0.01;
        //TODO Debug this
        try {
            BufferedReader b = new BufferedReader(new FileReader("tao_values.txt"));
            if(theta < 0.01)
                theta = 0.01;
            while(b.ready())
            {
                String [] line = b.readLine().split("\t");
                if(theta>=Double.parseDouble(line[0]) && theta <= (Double.parseDouble(line[0])+thetaInc))
                    return Double.parseDouble(line[boundIndex+1]);
            }
            System.err.println("Theta out of bounds");
            System.exit(-1);
            return -1;
        }
        catch(Exception e)
        {
            System.err.println("Exception in computing tao");
            e.printStackTrace();
            return -1;
        }
    }
    private double computeTao()
    {
        final double thetaInc = 0.01;
        //TODO Debug this
        try {
            BufferedReader b = new BufferedReader(new FileReader("tao_values.txt"));
            if(theta < 0.01)
                theta = 0.01;
            while(b.ready())
            {
                String [] line = b.readLine().split("\t");
                if(theta>=Double.parseDouble(line[0]) && theta <= (Double.parseDouble(line[0])+thetaInc))
                    return Double.parseDouble(line[boundIndex+1]);
            }
            System.err.println("Theta out of bounds");
            System.exit(-1);
            return -1;
        }
        catch(Exception e)
        {
            System.err.println("Exception in computing tao");
            e.printStackTrace();
            return -1;
        }
    }
    public double [][] addToMat(Graph g, DataSet data, double [][] edgeCounts)
    {
        for(Edge e:g.getEdges())
        {
            int x = data.getColumn(data.getVariable(e.getNode1().getName()));
            int y = data.getColumn(data.getVariable(e.getNode2().getName()));
            edgeCounts[x][y]++;
            edgeCounts[y][x]++;
        }
        return edgeCounts;

    }
}