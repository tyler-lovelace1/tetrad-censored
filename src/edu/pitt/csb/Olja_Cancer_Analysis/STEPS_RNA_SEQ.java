package edu.pitt.csb.Olja_Cancer_Analysis;

import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DelimiterType;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.search.CpcStable;
import edu.cmu.tetrad.search.IndependenceTest;
import edu.pitt.csb.mgm.IndTestMultinomialAJ;
import edu.pitt.csb.mgm.MGM;
import edu.pitt.csb.mgm.MixedUtils;
import edu.pitt.csb.mgm.STEPS;
import edu.pitt.csb.stability.SearchWrappers;
import edu.pitt.csb.stability.StabilityUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * This class can actually be used for both Microarray analysis and RNA-Seq analysis
 *  It will run STEPS and PC Stable on the full dataset to get neighbor estimates for the specified target
 *  It will also run a Leave one out Full Cross Validation (STEPS and PCS on each training dataset)
 */
public class STEPS_RNA_SEQ {
    public static void main(String [] args)throws Exception
    {
        String stepsFile = ""; //Allows reloading of the work that has already been done to avoid extra STEPS runs
        DelimiterType d2 = DelimiterType.TAB;
        int iterLimit = 100;
        DataSet d = MixedUtils.loadDataSet2(args[0],d2);
        String graphOut = args[1];
        String stabOut = args[2];
        String subDir = args[3];
        String target = args[4];
        boolean fullData = false;
        for(int i = 5; i < args.length;i++)
        {
            d.removeColumn(d.getVariable(args[i]));
        }



        File f = new File(subDir);
        if(!f.exists())
            f.mkdir();
        PrintStream pi1 = new PrintStream(subDir + "/Predictions.txt");
        PrintStream pi2 = new PrintStream(subDir + "/Features.txt");
        pi1.println("Run\tPrediction_MGM\tPrediction_CPC\tPrediction_CPC_MB\tActual");
        pi2.println("Run\tFeatures_MGM\tFeatures_CPC\tFeatures_CPC_MB");
       /* d.removeColumn(d.getVariable("Response"));
        d.removeColumn(d.getVariable("IgG_Week0"));
        d.removeColumn(d.getVariable("IgG_Week12"));
        d.removeColumn(d.getVariable("Response_DP"));*/
       // d.removeColumn(d.getVariable("IgG_Ratio"));
       // d.removeColumn(d.getVariable("Gender"));
       // d.removeColumn(d.getVariable("Age"));
        //  DataSet d3 = MixedUtils.completeCases(d);

System.out.println(d);
        //ns set to number of samples since we're doing leave-one-out cross validation
        int ns = 10;
        double g = 0.02;
        int numLambdas = 10;
        double [] lambda = new double[numLambdas];
        for(int i = 0; i < numLambdas;i++)
        {
            lambda[i] = .1+(.7/numLambdas)*i;
        }
        double [] l = null;
        double [][] stab = null;
        Graph g2 = null;
        BufferedReader b = null;
        if(fullData) {
            if (!stepsFile.equals("")) {
                b = new BufferedReader(new FileReader(stepsFile));

                String line = "";
                while (true) {
                    line = b.readLine();
                    if (line.startsWith("Lambdas"))
                        break;
                }

                String[] stuff = line.split("\\[")[1].split(",");
                l = new double[]{Double.parseDouble(stuff[0]), Double.parseDouble(stuff[1].trim()), Double.parseDouble(stuff[2].replace("]", "").trim())};
                MGM m = new MGM(d, l);
                m.learnEdges(iterLimit);
                g2 = m.graphFromMGM();
                stab = StabilityUtils.StabilitySearchPar(d, new SearchWrappers.MGMWrapper(lambda)).toArray();
            } else {
                STEPS s = new STEPS(d, lambda, g, ns, false);
                s.setComputeStabs(true);
                s.setIterLimit(iterLimit);
                System.out.print("Running StEPS...");
                g2 = s.runStepsPar();
                System.out.println("Done");
                stab = s.stabilities;
                l = s.lastLambda;
            }
        }       // System.out.println("Lambdas Chosen: " + Arrays.toString(l));


        //This first segment tells us what the model will look like using the full dataset
        //double [] params = {l[0],l[1],l[2]};
        //DataGraphSearch gs = new SearchWrappers.MGMWrapper(params);
        int [][] samps = StabilityUtils.generateSubsamples(d.getNumRows());
        DataSet [] subs = new DataSet[d.getNumRows()];
        for(int i = 0; i < d.getNumRows();i++) {
            subs[i] = d.copy();
            Arrays.sort(samps[i]);
            subs[i] = subs[i].subsetRows(samps[i]);
            PrintStream temp = new PrintStream(subDir + "/Test_Set_" + i + ".txt");
            for(int j = 0; j < d.getNumRows();j++)
            {
                if(Arrays.binarySearch(samps[i],j)<0)
                    temp.println(j+1);
            }
            temp.flush();
            temp.close();
            temp = new PrintStream(subDir + "/Training_Set_" + i + ".txt");
            for(int j = 0; j < samps[i].length;j++)
                temp.println(samps[i][j] + 1);
            temp.flush();
            temp.close();
        }

        if(fullData) {
            PrintStream out = new PrintStream(args[1]);
            out.println(g2);
            IndependenceTest iTest = new IndTestMultinomialAJ(d, 0.05);
            CpcStable pcs = new CpcStable(iTest);
            pcs.setInitialGraph(g2);
            //Graph g3 = pcs.search();
            out.flush();
            out.close();
            out = new PrintStream(stabOut);
            for (int i = 0; i < d.getNumColumns(); i++) {
                out.print(d.getVariable(i).getName());
                if (i < d.getNumColumns() - 1)
                    out.print("\t");
                else
                    out.println();
            }
            for (int i = 0; i < d.getNumColumns(); i++) {
                out.print(d.getVariable(i).getName() + "\t");
                for (int j = 0; j < d.getNumColumns(); j++) {
                    out.print(stab[i][j]);
                    if (j < d.getNumColumns() - 1)
                        out.print("\t");
                    else
                        out.println();
                }
            }

            out = new PrintStream("PCS_" + graphOut);
            //out.println(g3);
            out.flush();
            out.close();
        }

        //This portion tells us how good the modeling procedure is on this dataset (cross-validation of the full procedure)
try {
    for (int i = 0; i < subs.length; i++) {

        System.out.println("Cross Validation for Subsample " + i + " out of " + subs.length + "...");
        //Only select the features that are stable across subsamples, and that are connected when running MGM on full train
        DataSet train = subs[i];


        double[] lam = null;
        double[][] stabs = null;
        Graph gOut = null;
        System.out.print("Loading Lambdas...");
        if (!stepsFile.equals("")) {
            String line = "";
            while (b.ready()) {
                line = b.readLine();
                if (line.startsWith("Lambdas"))
                    break;
            }
            if (line.startsWith("Lambdas")) {
                String[] stuff = line.split("\\[")[1].split(",");
                lam = new double[]{Double.parseDouble(stuff[0]), Double.parseDouble(stuff[1].trim()), Double.parseDouble(stuff[2].replace("]", "").trim())};
                System.out.print("Using given lambda: " + Arrays.toString(lam) + "...");
                MGM m = new MGM(train, lam);
                m.learnEdges(1000);
                System.out.println("Done");
                gOut = m.graphFromMGM();
                stabs = StabilityUtils.StabilitySearchPar(train, new SearchWrappers.MGMWrapper(lam)).toArray();

            }

        }

        System.out.println("Done");


        if (lam == null) {
            System.out.print("Lambda is null so running steps...");
            STEPS s = new STEPS(train, lambda, g, ns, false);
            s.setComputeStabs(true);
            s.setIterLimit(iterLimit);
            gOut = s.runStepsPar();
            stabs = s.stabilities;
            lam = new double[]{s.lastLambda[0], s.lastLambda[1], s.lastLambda[2]};
            System.out.println("Done");
        }

        PrintStream p = new PrintStream(subDir + "/Neighbors_" + i + ".txt");
        PrintStream p2 = new PrintStream(subDir + "/Neighbors_PCS_" + i + ".txt");
        PrintStream p3 = new PrintStream(subDir + "/Neighbors_PCS_Causes_" + i + ".txt");


        System.out.print("Running PCS...");
        IndependenceTest indy = new IndTestMultinomialAJ(train, 0.05);
        CpcStable pc = new CpcStable(indy);
        pc.setInitialGraph(gOut);
        Graph gOut2 = pc.search();
        System.out.println("Done");

        List<Node> neighbors = new ArrayList<Node>();
        //MGM Adjacencies
        for (Node n : gOut.getAdjacentNodes(gOut.getNode(target))) {
            int x = d.getColumn(d.getVariable(n.getName()));
            int y = d.getColumn(d.getVariable(target));
            neighbors.add(n);
            p.println(n.getName() + "\t" + stabs[x][y]);
        }

        pi2.print(i + "\t" + neighbors + "\t");
        double res = PriorPrediction.getRegressionResult(neighbors,train,d,i,target);
        double actual = d.getDouble(i,d.getColumn(d.getVariable(target)));
        pi1.print(i + "\t" + res + "\t");

        List<Node> neighs = new ArrayList<Node>();
//PCS Adjacencies
        for (Node n : gOut2.getAdjacentNodes(gOut2.getNode(target))) {
            int x = d.getColumn(d.getVariable(n.getName()));
            int y = d.getColumn(d.getVariable(target));
            neighs.add(n);
            p2.println(n.getName() + "\t" + stabs[x][y]);
        }
        if(neighs.size()==0)
            neighs = neighbors;
            res = PriorPrediction.getRegressionResult(neighs,train,d,i,target);
        pi1.print(res + "\t");
        pi2.print(neighs + "\t");

        //PCS Markov Blanket Only
        neighs = gOut2.getAdjacentNodes(gOut2.getNode(target));
        for (Node n : gOut2.getChildren(gOut2.getNode(target))) {
            for(Node pp: gOut2.getParents(n))
            {
                if(!neighs.contains(pp))
                {
                    neighs.add(pp);
                    int x = d.getColumn(d.getVariable(pp.getName()));
                    int y = d.getColumn(d.getVariable(target));
                    p3.println(n.getName() + "\t" + stabs[x][y]);
                }
            }

        }
        if(neighs.size()==0)
            neighs = neighbors;
        res = PriorPrediction.getRegressionResult(neighs,train,d,i,target);
pi1.println(res + "\t" + actual);
pi2.println(neighs);

        p.flush();
        p.close();
        pi1.flush();
        pi2.flush();

        System.out.println("Done");
    }
    pi1.close();
    pi2.close();
}
catch(Exception e)
{
    System.out.println(e.getMessage());
    e.printStackTrace();
    System.exit(-1);
}
    }
}
