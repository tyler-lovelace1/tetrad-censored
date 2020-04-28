package edu.pitt.csb.latents;

import edu.cmu.tetrad.algcomparison.simulation.MixedLeeHastieSimulation;
import edu.cmu.tetrad.algcomparison.simulation.Parameters;
import edu.cmu.tetrad.data.ContinuousVariable;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DiscreteVariable;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.GraphUtils;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.graph.NodeType;
import edu.cmu.tetrad.search.DagToPag;
import edu.pitt.csb.mgm.MixedUtils;
import edu.pitt.csb.stability.Bootstrap;
import edu.pitt.csb.stability.CPSS;
import edu.pitt.csb.stability.StabilityUtils;
import org.apache.commons.math3.distribution.NormalDistribution;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by vinee_000 on 10/9/2017.
 */
    //TODO Modify this script to save and load these partitions properly for the CPSS methods
public class testLatentRecovery {
    public static void main(String [] args)throws Exception
    {
        boolean numEdgesRandom = true;
        boolean prune = false; // prune latent predictions that don't hold up to the explain away test
        boolean requireFullEdge = false; //Require the edge to be present in the full dataset to count
        boolean noDiscreteLatents = true; //Don't allow discrete variables to be latents (no discrete variables in the TCGA)
        int numLambdas = 40;
        int numVariables = 50;
        int numLatents = 10;
        int numEdges = 200;
        int sampleSize = 200;
        int numSubsamples = 10;
        int numRuns = 15;
        int index = 0;
        int numCategories = 4;
        int BS = 100;
        //int numSubSets = numVariables/5;
        int numSubSets = 4;
        double percentDiscrete = 2;
        double tao = 0.1;
        double stepsGamma = 0.05;
        double starsGamma = 0.0001;
        boolean saveData = true;
        boolean reuseData = true;
        boolean rerunAlgorithms = true;
        int B = 50; //Number of partitions for CPSS
        double cpAlpha = 0.05;
        double [] cpLambda = {0.2,0.2,0.2};
        double bound = 0.05;
        double bootstrapBound = 0;
        String directory = ".";

        //String[] algs = {"FCI","MGM-FCI","MGM-FCI-MAX","Latent_FCI","Latent_MGM-FCI","Latent_MGM-FCI-MAX"};
//String [] algs = {"FCI","Latent_FCI","MGM-FCI-MAX","Latent_MGM-FCI-MAX"};
       // String [] algs = {"MGM-FCI-MAX","Latent_MGM-FCI-MAX"};
        //String [] algs= {"Latent_FCI","Latent_MGM-FCI-MAX"};
       // String [] algs = {"MGM-FCI-MAX","FCI"};
        String [] algs = {"FCI","MGM-FCI-MAX","CPSS-FCI","CPSS-MGM-FCI-MAX","Bootstrap-FCI","Bootstrap-MGM-FCI-MAX"};

      //  String [] algs = {"Bootstrap-FCI","Bootstrap-MGM-FCI-MAX"};

        //String [] algs = {"FCI","MGM-FCI","MGM-FCI-MAX","CPSS-FCI","CPSS-MGM-FCI","CPSS-MGM-FCI-MAX"};
        double [][][] precision = new double[numRuns][algs.length][4];
        double [][][] recall = new double[numRuns][algs.length][4];
        double [][][] identRecall = new double[numRuns][algs.length][4];
        double [][] runtimes = new double[numRuns][algs.length];
        String [] types = {"CC","CD","DD","All"};
        while(index < args.length)
        {
            if(args[index].equals("-rd"))
            {
                reuseData = true;
                index++;
            }
            else if(args[index].equals("-d")) {
                directory = args[index + 1];
                index += 2;
            }
            else if(args[index].equals("-nl"))
            {
                numLatents = Integer.parseInt(args[index+1]);
                index+=2;
            }
            else if(args[index].equals("-nc"))
            {
                numCategories = Integer.parseInt(args[index+1]);
                index+=2;
            }
            else if(args[index].equals("-sv"))
            {
                saveData = true;
                index++;
            }
            else if(args[index].equals("-l"))
            {
                numLambdas = Integer.parseInt(args[index+1]);
                index+=2;
            }
            else if(args[index].equals("-v"))
            {
                numVariables = Integer.parseInt(args[index+1]);
                index+=2;
            }
            else if(args[index].equals("-e"))
            {
                numEdges = Integer.parseInt(args[index+1]);
                index+=2;
            }
            else if(args[index].equals("-s"))
            {
                sampleSize = Integer.parseInt(args[index+1]);
                index+=2;
            }
            else if(args[index].equals("-ns"))
            {
                numSubsamples = Integer.parseInt(args[index+1]);
                index+=2;
            }
            else if(args[index].equals("-r"))
            {
                numRuns = Integer.parseInt(args[index+1]);
                index+=2;
            }
            else if(args[index].equals("-sg"))
            {
                stepsGamma = Double.parseDouble(args[index+1]);
                index+=2;
            }
            else if(args[index].equals("-*g"))
            {
                starsGamma = Double.parseDouble(args[index+1]);
                index+=2;
            }
        }

        boolean cpss = false;
        for(String s:algs)
            if(s.contains("CPSS"))
                cpss = true;

        double [] initLambdas = new double[numLambdas];
        double low = .05;
        double high = .9;
        double inc = (high-low)/(numLambdas-1);
        for(int i = 0; i < numLambdas;i++)
        {
            initLambdas[i] = i*inc + low;
        }
        File rFile = new File("Results");
        if(!rFile.isDirectory())
            rFile.mkdir();
        File gFile = new File("Graphs");
        File dFile = new File("Data");
        File pFile = new File("Priors");
        File estFile = new File("Estimated");
        File egFile = new File("Estimated Graphs");
        File subFile = new File("Subsamples");
        File partFile = new File("Partitions");
        File runFile = new File("Runtimes");
        File bootFile = new File("Bootstraps");
        if(saveData) {
            if (!gFile.isDirectory())
                gFile.mkdir();
            if (!dFile.isDirectory())
                dFile.mkdir();
            if(!estFile.isDirectory())
                estFile.mkdir();
            if(!pFile.isDirectory())
                pFile.mkdir();
            if(!subFile.isDirectory())
                subFile.mkdir();
            if(!egFile.isDirectory())
                egFile.mkdir();
            if(!runFile.isDirectory())
                runFile.mkdir();
            if(!partFile.isDirectory())
                partFile.mkdir();
            if(!bootFile.isDirectory())
                bootFile.mkdir();
        }

        /*PrintStream pri = new PrintStream(directory + "/mgm_priors_" + amountPrior + "_" + numExperts + "_" + numVariables +  "_" + sampleSize + "_" + numSubsamples + ".txt");
        PrintStream step = new PrintStream(directory + "/STEPS_" + amountPrior + "_" + numExperts + "_" + numVariables + "_"  + sampleSize + "_" + numSubsamples + ".txt");
        PrintStream orc = new PrintStream(directory + "/oracle_" + amountPrior + "_" + numExperts + "_" + numVariables + "_" + sampleSize + "_" + numSubsamples + ".txt");
        PrintStream one = new PrintStream(directory + "/mgm_one_steps_" + amountPrior + "_" + numExperts + "_" + numVariables + "_" + sampleSize + "_" + numSubsamples + ".txt");
        PrintStream orcOne = new PrintStream(directory + "/oracle_one_" + amountPrior + "_" + numExperts + "_" + numVariables + "_" + sampleSize + "_" + numSubsamples + ".txt");*/
        NormalDistribution n = new NormalDistribution(numVariables*2,numVariables/2);
        PrintStream [] pri = new PrintStream[types.length];
        PrintStream pri2 = new PrintStream("Runtimes/Runtime_" + numVariables + "_" + sampleSize + "_" + numLatents + "_" + numRuns + "_" + numSubSets +  ".txt");
        for(int i = 0; i < types.length;i++) {
            pri[i] = new PrintStream("Results/result_" + numVariables + "_" + sampleSize + "_" + numLatents + "_" + numRuns + "_" + types[i] +  ".txt");
        }
        A:for(int i = 0; i < numRuns; i++) {
            MixedLeeHastieSimulation c = new MixedLeeHastieSimulation();
            System.out.println(i);
            if(numEdgesRandom)
            {
                numEdges = (int)n.sample();
                while(numEdges <numVariables)
                    numEdges = (int)n.sample();
            }
            Parameters p = new Parameters();
            p.setValue("numMeasures", numVariables);
            p.setValue("numEdges", numEdges);
            p.setValue("sampleSize", sampleSize);
            p.setValue("numCategories",numCategories);
            p.setValue("numLatents",numLatents);
            p.setValue("percentDiscreteForMixedSimulation",percentDiscrete);
            c.simulate(p);
            while(noDiscreteLatents && badLatent(c.getDataSet(0),c.getTrueGraph()))
            {
                c.simulate(p);
            }
            int[][] subsamples = new int[numSubsamples][];
            int [][] partitions = new int[B*2][];
            int [][] bootParts = new int[BS][];
            if(reuseData)
            {
                boolean foundFile = false;
                boolean foundData = false;
                File f = new File("Graphs/Graph_" + i + "_" + numVariables + "_" + numLatents +  ".txt");
                if(f.exists()) {
                    foundFile = true;
                    c.setTrueGraph(GraphUtils.loadGraphTxt(f));
                }
                f = new File("Data/Data_" + i + "_" + numVariables + "_" + sampleSize + "_" + numLatents + ".txt");
                if(f.exists() && foundFile) {
                    foundData = true;
                    c.setDataSet(MixedUtils.loadDataSet2("Data/Data_" + i + "_" + numVariables + "_" + sampleSize + "_" + numLatents + ".txt"), 0);
                    Graph t = c.getTrueGraph();
                    for(int x = 1; x <= numVariables;x++)
                    {
                        if(c.getDataSet(0).getVariable("X" + x)==null)
                            t.getNode("X" + x).setNodeType(NodeType.LATENT);
                    }
                    c.setTrueGraph(t);

                }
                else {
                    while(noDiscreteLatents && badLatent(c.getDataSet(0),c.getTrueGraph()))
                    {
                        c.simulate(p);
                    }

                    Graph t = c.getTrueGraph();
                    for(int x = 1; x <= numVariables;x++)
                    {
                        if(c.getDataSet(0).getVariable("X" + x)==null)
                            t.getNode("X" + x).setNodeType(NodeType.LATENT);
                    }
                }
                if(foundData) {

                        f = new File("Subsamples/Subsample_" + i + "_" + numVariables + "_" + sampleSize + "_" + numLatents + ".txt");

                        if (f.exists()) {
                            BufferedReader b2 = new BufferedReader(new FileReader(f.getAbsolutePath()));
                            for (int j = 0; j < numSubsamples; j++) {
                            String [] line = b2.readLine().split("\t");
                            subsamples[j] = new int[line.length];
                            for(int k = 0; k < line.length;k++)
                            {
                                subsamples[j][k] = Integer.parseInt(line[k]);
                            }

                        }
                    }
                    f = new File("Partitions/Partition_" + i + "_" + numVariables + "_" + sampleSize + "_" + numLatents + ".txt");
                        if(f.exists()) {
                            BufferedReader b2 = new BufferedReader(new FileReader(f.getAbsolutePath()));
                            for (int j = 0; j < B * 2; j++) {
                                String[] line = b2.readLine().split("\t");
                                partitions[j] = new int[line.length];
                                for (int k = 0; k < line.length; k++) {
                                    partitions[j][k] = Integer.parseInt(line[k]);
                                }

                            }
                            b2.close();
                        }

                    f = new File("Bootstraps/Bootstraps_" + i + "_" + numVariables + "_" + sampleSize + "_" + numLatents + ".txt");
                    if(f.exists()) {
                        BufferedReader b2 = new BufferedReader(new FileReader(f.getAbsolutePath()));
                        for (int j = 0; j < BS; j++) {
                            String[] line = b2.readLine().split("\t");
                            bootParts[j] = new int[line.length];
                            for (int k = 0; k < line.length; k++) {
                                bootParts[j][k] = Integer.parseInt(line[k]);
                            }

                        }
                        b2.close();
                    }

                }

            }
            ArrayList<ArrayList<LatentPrediction.Pair>> estimatedGraphs = new ArrayList<ArrayList<LatentPrediction.Pair>>();
            ArrayList<Map<String,String>> orientations = new ArrayList<Map<String,String>>();

            ArrayList<int[][]> splits = new ArrayList<int[][]>();
            for(int j = 0; j < algs.length;j++)
            {
                estimatedGraphs.add(null);
                orientations.add(null);
                splits.add(null);
            }
            if(!rerunAlgorithms) {
                for (int j = 0; j < algs.length; j++) {
                    File f = new File("Estimated/" + algs[j] + "_" + i + "_" + numVariables + "_" + sampleSize + "_" + numLatents + "_" + numSubSets + ".txt");

                    if (f.exists()) {

                        BufferedReader b = new BufferedReader(new FileReader("Estimated/" + algs[j] + "_" + i + "_" + numVariables + "_" + sampleSize + "_"  + numLatents + "_" + numSubSets + ".txt"));
                        ArrayList<LatentPrediction.Pair> temp = new ArrayList<LatentPrediction.Pair>();
                        while(b.ready())
                        {
                            String [] line = b.readLine().split("\t");
                            if(line[0].equals("Splits") || line[0].equals("Orientations"))
                                break;
                            LatentPrediction.Pair p2 = new LatentPrediction.Pair(new ContinuousVariable(line[0]),new ContinuousVariable(line[1]),Double.parseDouble(line[2]));
                            temp.add(p2);
                        }
                        //b.close();

                        if(algs[j].contains("Latent")) {
                            int[][] currSplit = new int[numSubSets][];
                            for (int m = 0; m < numSubSets; m++) {
                                String[] line = b.readLine().split("\t");
                                currSplit[m] = new int[line.length];
                                for (int k = 0; k < line.length; k++) {
                                    currSplit[m][k] = Integer.parseInt(line[k]);
                                }
                            }
                            splits.set(j,currSplit);
                        }

                        Map<String,String> t2 = new HashMap<String,String>();
                        b.readLine();
                        while(b.ready())
                        {
                            String [] line = b.readLine().split("\t");
                            String tempX = line[1];
                            for(int ii = 2; ii < line.length;ii++)
                                tempX+=("\t"+line[ii]);
                            t2.put(line[0],tempX);
                        }
                        b.close();
                        estimatedGraphs.set(j,temp);
                        orientations.set(j,t2);
                    }



                }
            }
            boolean done = false;
            while(!done)
            {
                System.out.println(i);
                try {
                    boolean nullSub = false;
                    boolean nullParts = false;
                    boolean nullBoots = false;
                    for(int j = 0; j < subsamples.length;j++)
                    {
                        if(subsamples[j]==null)
                        {
                            nullSub = true;
                        }
                    }
                    for(int j = 0; j < partitions.length;j++)
                    {
                        if(partitions[j]==null)
                            nullParts = true;
                    }
                    for(int j = 0; j < bootParts.length;j++)
                    {
                        if(bootParts[j]==null)
                            nullBoots = true;
                    }

                    if(nullParts)
                        partitions = CPSS.createSubs(c.getDataSet(0),B);
                    if(nullSub) {
                        int b = (int) Math.floor(10 * Math.sqrt(c.getDataSet(0).getNumRows()));
                        if (b > c.getDataSet(0).getNumRows())
                            b = c.getDataSet(0).getNumRows() / 2;
                        subsamples = StabilityUtils.subSampleNoReplacement(c.getDataSet(0).getNumRows(), b, numSubsamples);

                    }
                    if(nullBoots)
                    {
                        bootParts = Bootstrap.createSubs(c.getDataSet(0),BS);
                    }




                    for(int j = 0; j < algs.length;j++)
                    {
                        System.out.println("Running " + algs[j]);
                        if(!algs[j].contains("Latent"))
                        {
                            if(estimatedGraphs.get(j)==null) {
                                String temp = algs[j].replace("Latent_","");
                                LatentPrediction lp = new LatentPrediction(c.getDataSet(0),numSubSets,tao,subsamples);
                             //   lp.setAlpha(0.01);
                                if(algs[j].contains("CPSS"))
                                {
                                    temp = temp.replace("CPSS-","");
                                    lp.setCPSS(B,cpAlpha,cpLambda,bound,partitions);
                                }
                                if(algs[j].contains("Bootstrap"))
                                {
                                    temp = temp.replace("Bootstrap-","");
                                    lp.setBootstrap(BS,cpAlpha,cpLambda,bootstrapBound,bootParts);
                                }
                                lp.setStarsGamma(starsGamma);
                                lp.setStepsGamma(stepsGamma);
                                lp.setGraph(c.getTrueGraph());
                                lp.setRunNumber(i);
                                long time = System.nanoTime();

                                ArrayList<LatentPrediction.Pair> latents = lp.runRegularAlgorithm(temp,requireFullEdge);
                                System.out.println(latents);
                                runtimes[i][j] = (System.nanoTime()-time)/(Math.pow(10,9));
                                addNonPredictedLatents(latents, c.getTrueGraph(),c.getDataSet(0));
                                System.out.println(latents);
                                estimatedGraphs.set(j,latents);
                                orientations.set(j,lp.orientations);
                               // System.out.println(orientations.get(j));
                            }


                        }
                        else
                        {
                            if(estimatedGraphs.get(j)==null) {
                                String temp = algs[j].replace("Latent_","");
                                LatentPrediction lp = new LatentPrediction(c.getDataSet(0),numSubSets,tao,subsamples);
                                if(algs[j].contains("CPSS"))
                                {
                                    temp = temp.replace("CPSS-","");
                                    lp.setCPSS(B,cpAlpha,cpLambda,bound,partitions);
                                }
                                if(algs[j].contains("Bootstrap"))
                                {
                                    temp = temp.replace("CPSS-","");
                                    lp.setBootstrap(BS,cpAlpha,cpLambda,bootstrapBound,bootParts);
                                }
                                lp.setStarsGamma(starsGamma);
                                lp.setStepsGamma(stepsGamma);
                                lp.setGraph(c.getTrueGraph());
                                lp.setRunNumber(i);
                                long time = System.nanoTime();
                                ArrayList<LatentPrediction.Pair> latents = lp.runAlgorithm(temp,prune,requireFullEdge);
                                runtimes[i][j] = (System.nanoTime()-time)/Math.pow(10,9);

                                //TODO this doesn't take splits into account so need to note that
                                addNonPredictedLatents(latents, c.getTrueGraph(),c.getDataSet(0));
                                estimatedGraphs.set(j,latents);
                                splits.set(j,lp.getLastSplit());
                                orientations.set(j,lp.orientations);
                            }
                        }

                        System.out.println("Doing DAG to PAG conversion");
                        DagToPag pg = new DagToPag(c.getTrueGraph());
                        pg.setCompleteRuleSetUsed(false);
                        Graph truePag = pg.convert();
                        //TODO change this back to all types
                        for(int k = 3; k < types.length;k++)
                        {
                            if(!algs[j].contains("Latent")) {
                                double r = LatentPrediction.getRecall(estimatedGraphs.get(j), c.getTrueGraph(), c.getDataSet(0), types[k]);

                                recall[i][j][k] = r;
                                identRecall[i][j][k] = LatentPrediction.getIdentifiableRecall(truePag, estimatedGraphs.get(j),c.getTrueGraph(),c.getDataSet(0),types[k]);

                            }
                            else
                            {
                                double r = LatentPrediction.getRecall(estimatedGraphs.get(j), splits.get(j), c.getTrueGraph(), c.getDataSet(0),types[k]);
                                recall[i][j][k] = r;
                                identRecall[i][j][k] = LatentPrediction.getIdentifiableRecall(truePag, estimatedGraphs.get(j),c.getTrueGraph(),c.getDataSet(0),types[k],splits.get(j));
                            }
                            double pre = LatentPrediction.getPrecision(estimatedGraphs.get(j),c.getTrueGraph(),c.getDataSet(0),types[k]);
                            precision[i][j][k] = pre;
                            System.out.println(algs[j] + "," + pre);

                        }
                        //TOD

                    }

                    done = true;
                } catch (Exception e) {
                    e.printStackTrace();
                    System.exit(0);
                    if(!reuseData) {
                        c.simulate(p);
                    }
                }
            }

            if(saveData) {
                PrintStream p2 = new PrintStream("Graphs/Graph_" + i + "_" + numVariables + "_" + numLatents +   ".txt");
                p2.println(c.getTrueGraph());
                p2.flush();
                p2.close();
                p2 = new PrintStream("Data/Data_" + i + "_" + numVariables + "_" + sampleSize + "_" + numLatents + ".txt");
                p2.println(c.getDataSet(0));
                p2.flush();
                p2.close();
                for(int x = 0; x < algs.length;x++) {
                    p2 = new PrintStream("Estimated/" + algs[x] + "_" + i + "_" + numVariables + "_" + sampleSize + "_" + numLatents + "_" + numSubSets +  ".txt");
                    for (LatentPrediction.Pair mRNA : estimatedGraphs.get(x)) {
                        p2.println(mRNA.one.getName() + "\t" + mRNA.two.getName() + "\t" + mRNA.stability);
                    }
                    int[][] sam = splits.get(x);
                    if(sam !=null) {
                        p2.println("Splits");

                        for (int j = 0; j < sam.length; j++) {
                            for (int k = 0; k < sam[j].length; k++) {
                                if (k != sam[j].length - 1)
                                    p2.print(sam[j][k] + "\t");
                                else
                                    p2.println(sam[j][k]);
                            }
                        }
                    }
                    p2.println("Orientations");
                    Map<String,String> temp = orientations.get(x);
                    for(String y:temp.keySet())
                    {
                        p2.println(y + "\t" + temp.get(y));
                    }
                    p2.flush();
                    p2.close();
                }
                p2 = new PrintStream("Subsamples/Subsample_" + i + "_" + numVariables + "_" + sampleSize + "_" + numLatents + ".txt");
                for(int j = 0; j < subsamples.length;j++)
                {
                    for(int k = 0; k < subsamples[j].length;k++)
                    {
                        p2.print(subsamples[j][k] + "\t");
                    }
                   p2.println();
                }
                p2.flush();
                p2.close();
                p2 = new PrintStream("Partitions/Partition_" + i + "_" + numVariables + "_" + sampleSize + "_" + numLatents + ".txt");
                for(int j = 0; j < partitions.length;j++)
                {
                    for(int k = 0; k < partitions[j].length;k++)
                    {
                        p2.print(partitions[j][k] + "\t");
                    }
                    p2.println();
                }
                p2.flush();
                p2.close();
                p2 = new PrintStream("Bootstraps/Bootstrap_" + i + "_" + numVariables + "_" + sampleSize + "_" + numLatents + ".txt");
                for(int j = 0; j < bootParts.length;j++)
                {
                    for(int k = 0; k < bootParts[j].length;k++)
                    {
                        p2.print(bootParts[j][k] + "\t");
                    }
                    p2.println();
                }
                p2.flush();
                p2.close();
            }
        }
        for(int i = 0; i < types.length;i++)
        printData(pri[i],precision,recall,identRecall,algs,i);

        printRuntimes(pri2,runtimes,algs);
    }
    public static void printRuntimes(PrintStream out, double [][] runtimes,String [] algs)
    {
        out.print("Run\t");
        for(int j = 0; j < algs.length;j++)
        {
            if(j==algs.length-1)
                out.println(algs[j]);
            else
                out.print(algs[j] + "\t");
        }

        for(int i = 0; i < runtimes.length;i++)
        {
            out.print(i + "\t");
            for(int j = 0; j < algs.length;j++)
            {
                if(j==algs.length-1)
                    out.println(runtimes[i][j]);
                else
                    out.print(runtimes[i][j] + "\t");
            }
        }
        out.flush();
        out.close();

    }
    public static void printData(PrintStream out, double [][][] precision, double [][][] recall, double [][][] identRecall,String [] algs, int type)throws Exception
    {
        out.print("Run\t");
        for(int j = 0; j < algs.length;j++)
        {
            if(j!=algs.length-1)
                out.print(algs[j] + "_PREC\t" + algs[j] + "_REC\t" + algs[j] + "_IREC\t");
            else
                out.println(algs[j] + "_PREC\t" + algs[j] + "_REC\t" + algs[j] + "_IREC");
        }
        for(int j = 0; j < precision.length;j++)
        {
            out.print(j + "\t");
            for(int k = 0; k < algs.length;k++)
            {
                if(k!=algs.length-1)
                {
                    out.print(precision[j][k][type] + "\t" + recall[j][k][type] + "\t" + identRecall[j][k][type] + "\t");
                }
                else
                {
                    out.println(precision[j][k][type] + "\t" + recall[j][k][type] + "\t" + identRecall[j][k][type]);
                }
            }
        }
        out.flush();
        out.close();
    }

    private static void addNonPredictedLatents(List<LatentPrediction.Pair> latents, Graph trueGraph, DataSet data)
    {
        List<LatentPrediction.Pair> allLatents = LatentPrediction.getLatents(trueGraph,data,"All");
        for(int i = 0; i < allLatents.size();i++)
        {
            boolean found = false;
            for(int j = 0; j < latents.size();j++ )
            {
                if((latents.get(j).colIDOne==allLatents.get(i).colIDOne && latents.get(j).colIDTwo==allLatents.get(i).colIDTwo) || (latents.get(j).colIDTwo==allLatents.get(i).colIDOne && latents.get(j).colIDOne==allLatents.get(i).colIDTwo))
                {
                   found = true;
                }
            }
            if(found)
                latents.add(new LatentPrediction.Pair(data.getVariable(allLatents.get(i).one.getName()),data.getVariable(allLatents.get(i).two.getName()),0.0));
        }
    }
    private static boolean badLatent(DataSet d, Graph g)
    {
        //Returns true if the Graph g has a discrete variable as a latent or if a discrete variable is the child of a latent
        for(Node n: g.getNodes())
        {
            if(n.getNodeType()==NodeType.LATENT)
            {
                if(d.getVariable(n.getName()) instanceof DiscreteVariable)
                {
                    return true;
                }
                for(Node n2: g.getChildren(n))
                {
                    if(d.getVariable(n2.getName())instanceof DiscreteVariable)
                        return true;
                }
            }
        }
        return false;
    }
}
