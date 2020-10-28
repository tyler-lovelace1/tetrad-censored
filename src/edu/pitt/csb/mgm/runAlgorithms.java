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

package edu.pitt.csb.mgm;

import edu.cmu.tetrad.data.*;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.*;
import edu.pitt.csb.stability.Bootstrap;
import edu.pitt.csb.stability.DataGraphSearch;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by Vineet Raghu on 9/11/2017
 * The purpose of this class is to create a smooth interface to run several causal discovery algorithms suitable for mixed discrete and continuous data
 * This class has been compiled into a .jar file for efficient use
 * TODO allow input of true graph in order to estimate typical orientation and adjacency problems
 * TODO Allow for bootstrapping of the entire causal process
 * TODO Allow choice of independence test
 */
public class runAlgorithms {
    private static Graph trueGraph = null; //TODO
    private static String dataFile = ""; //Data file to be used by the algorithms
    private static DataSet d = null; //The loaded dataset itself, must be specified by the user
    private static double [] lambda = {.2,.2,.2}; //Array of Lambda values for MGM
    private static double [] survlambda = {.2,.2,.2,.2,.2}; //Array of Lambda values for MGM
    private static int count = 0; //Upkeep variable to load parameters
    private static String alg = "None"; //Name of the algorithm to run
    private static double alpha = .05; //Independence test threshold for constraint-based causal discovery algorithms
    private static String outputPath = "./output.txt"; //Output file for the learned causal graph
    private static boolean outputSif = false; //Should we output an .sif format file
    private static String sifPath = ""; //Path to the .sif format file output
    private static boolean runSteps = false; //Should we run steps?
    private static int ns = 20; //Number of subsamples for StARS/StEPS
    private static int b = -1; //Subsample size, will be computed automatically later as 10*sqrt(sample size)
    private static double threshold = 0.05; //Stability threshold for StARS/StEPS
    private static boolean useKnowledge = false; //Should we use a tetrad knowledge file?
    private static String kFile = "";//Path to the tetrad knowledge file
    private static double penalty = 1; //Penalty discount/Structure Prior for Score for FGES
    private static int maxNumDiscrete = 5; //Maximum number of categories a variable can have to be considered a discrete variable
    private static boolean runStars = false; //Should we run StARS?
    private static boolean runMGM = false;// Should we run MGM to get an undirected skeleton?
    private static boolean runsurvivalMGM = false;// Should we run survivalMGM to get an undirected skeleton?
    private static boolean solvePath = false;// Should we get the solution path for an array of lambda values
    private static int numParams = 20; //Number of parameters to test for StARS/StEPS
    private static boolean runCPSS = false; //Should we run CPSS?
    private static double cpssBound = 0.05; //Error rate threshold for CPSS
    private static boolean runBootstrap = false; //Should we run bootstrapping?
    private static double bootBound = 50; //Number of times an orientation must appear to be added to the output
    private static int numBoots = 100; //Number of bootstrapped samples to run
    private static boolean outputStabs = false; //Should we output edge stabilities (StARS/StEPS)?
    private static String stabPath = ""; //Path to stability file
    private static boolean outputOrients = false; //Should we output orientation frequencies for CD algorithms or CPSS or Bootstrapping?
    private static String orientPath = ""; //Path to the orientation frequency file
    private static double paramLow = -1; //Lov value of parameter range to test for StEPS
    private static double paramHigh = -1; //High value of parameter range to test for StEPS
    private static double paramMult = -1; //Multiplier for log-linear spaced test parameters for StEPS
    private static double paramLowStars = -1;//Lov value of parameter range to test for StARS
    private static double paramHighStars = -1;//High value of parameter range to test for StARS
    private static Graph initGraph; //Initial Undirected Skeleton to use before running causal discovery
    private static ArrayList<String> toRemove = new ArrayList<String>();
    public static void main(String[] args) throws Exception{
        try {

            //Interpet command line arguments
            while(count < args.length)
            {
                if(args[count].equals("-k")) //Knowledge file to use in Tetrad knowledge format
                {
                    useKnowledge = true;
                    kFile = args[count+1];
                    count+=2;
                }
                else if(args[count].equals("-orientStabs")) //File to output orientation stability
                {
                    outputOrients = true;
                    orientPath = args[count+1];
                    count+=2;
                }
                else if(args[count].equals("-stars")) //Should we run StARS? Only for causal (directed) search algorithms
                {
                    runStars = true;
                    if(args.length==count+1 || args[count+1].startsWith("-")) {

                        count++;
                    }
                    else //Optional: how many parameters should we test?
                    {
                        numParams = Integer.parseInt(args[count+1]);
                        count+=2;
                    }

                }
                else if(args[count].equals("-cpss"))
                {
                    runCPSS = true;
                    if(args.length==count+1||args[count+1].startsWith("-"))
                        count++;
                    else {
                        cpssBound = Double.parseDouble(args[count + 1]);
                        count += 2;
                    }
                }
                else if(args[count].equals("-initGraph"))
                {
                    initGraph = GraphUtils.loadGraphTxt(new File(args[count+1]));
                    count+=2;
                }
                else if(args[count].equals("-bootstrap"))
                {
                    runBootstrap = true;
                    count++;
                }
                else if(args[count].equals("-numBoots"))
                {
                    numBoots = Integer.parseInt(args[count+1]);
                    count+=2;
                }
                else if(args[count].equals("-bootBound"))
                {
                    bootBound = Double.parseDouble(args[count+1]);
                    count+=2;
                }
                else if(args[count].equals("-low")) //Low value of parameter range to test (for StARS/StEPS)
                {
                    paramLow = Double.parseDouble(args[count+1]);
                    count+=2;
                }
                else if(args[count].equals("-lowStars"))
                {
                    paramLowStars = Double.parseDouble(args[count+1]);
                    count+=2;
                }
                else if(args[count].equals("-high")) //High value of parameter range to test
                {
                    paramHigh = Double.parseDouble(args[count+1]);
                    count+=2;
                }
                else if(args[count].equals("-highStars"))
                {
                    paramHighStars = Double.parseDouble(args[count+1]);
                    count+=2;
                }
                else if(args[count].equals("-mult")) //Low value of parameter range to test (for StARS/StEPS)
                {
                    paramMult = Double.parseDouble(args[count+1]);
                    count+=2;
                }
                else if(args[count].equals("-b")) //Subsample size for StARS/StEPS
                {
                    b = Integer.parseInt(args[count+1]);
                    count+=2;
                }
                else if(args[count].equals("-mgm"))
                {
                    runMGM = true;
                    count++;
                }
                else if(args[count].equals("-survivalmgm"))
                {
                    runsurvivalMGM = true;
                    count++;
                }
                else if(args[count].equals("-path"))
                {
                    solvePath = true;
                    if(args.length==count+1 || args[count+1].startsWith("-")) {
                        count++;
                    }
                    else //Optional: how many parameters should we test?
                    {
                        numParams = Integer.parseInt(args[count+1]);
                        count+=2;
                    }
                }
                else if(args[count].equals("-maxCat")) //Maximum number of categories any discrete variable has
                {
                    //Maximum number of categories for a discrete variable
                    maxNumDiscrete = Integer.parseInt(args[count+1]);
                    if(maxNumDiscrete <= 0)
                    {
                        System.err.println("Maximum number of categories for discrete variable cannot be negative");
                        System.exit(-1);
                    }
                    count+=2;
                }
                else if(args[count].equals("-steps")) //Should steps be run?
                {
                    runSteps = true;
                    // runMGM = true;
                    if(args.length==count+1 || args[count+1].startsWith("-")) {
                        count++;
                    }
                    else //Optional: how many parameters should we test?
                    {

                        numParams = Integer.parseInt(args[count+1]);
                        count+=2;
                    }
                }
                else if(args[count].equals("-g")) //Stability threshold for StEPS/StARS
                {
                    threshold = Double.parseDouble(args[count+1]);
                    count+=2;
                }
                else if(args[count].equals("-ns")) //Number of subsamples for StEPS/StARS
                {
                    ns = Integer.parseInt(args[count+1]);
                    if(ns <=0)
                    {
                        System.err.println("Number of Subsamples must be greater than zero: " + ns);
                        System.exit(-1);
                    }
                    count+=2;
                }
                else if(args[count].equals("-sif"))//Should we output a file in a format suitable for cytoscape
                {
                    outputSif = true;
                    sifPath = args[count+1];
                    count+=2;
                }
                else if(args[count].equals("-stabs")) //Should we output adjacency stabilities?
                {
                    outputStabs = true;
                    if(args.length==count+1 || args[count+1].startsWith("-"))
                    {
                        System.err.println("Must specify a file name to output the adjacency stabilities");
                        System.exit(-1);
                    }
                    stabPath = args[count+1];
                    count+=2;
                }
               else if(args[count].equals("-d"))  //Required: dataset filepath
               {
                    dataFile = args[count + 1];
                    count+=2;
                }
                else if(args[count].equals("-o")) //Path for output file for general tetrad graph output
                {
                    outputPath = args[count+1];
                    count+=2;
                }
                else if(args[count].equals("-l"))  //Lambda parameters for MGM
                {
                    if (!runsurvivalMGM) {
                        lambda[0] = Double.parseDouble(args[count + 1]);
                        lambda[1] = Double.parseDouble(args[count + 2]);
                        lambda[2] = Double.parseDouble(args[count + 3]);
                        for (int i = 0; i < lambda.length; i++) {
                            if (lambda[i] < 0 || lambda[i] > 1) {
                                System.err.println("Invalid value for lambda[" + i + "], " + lambda[i]);
                                System.exit(-1);
                            }
                        }
                        count += 4;
                    }
                    else {
                        survlambda[0] = Double.parseDouble(args[count + 1]);
                        survlambda[1] = Double.parseDouble(args[count + 2]);
                        survlambda[2] = Double.parseDouble(args[count + 3]);
                        survlambda[3] = Double.parseDouble(args[count + 4]);
                        survlambda[4] = Double.parseDouble(args[count + 5]);
                        for (int i = 0; i < survlambda.length; i++) {
                            if (survlambda[i] < 0 || survlambda[i] > 1) {
                                System.err.println("Invalid value for lambda[" + i + "], " + survlambda[i]);
                                System.exit(-1);
                            }
                        }
                        count += 6;
                    }
                }
                else if(args[count].equals("-alg")) //Which algorithm should be used
                {
                    alg = args[count+1];
                    count+=2;
                }
                else if(args[count].equals("-a")) //Alpha value for constraint-based causal discovery algorithms
                {
                    alpha = Double.parseDouble(args[count+1]);
                    if(alpha < 0 || alpha > 1)
                    {
                        System.err.println("Invalid value for alpha = " + alpha);
                        System.exit(-1);
                    }
                    count+=2;
                }
                else if(args[count].equals("-penalty")) //Penalty for score-based causal discovery algorithms
                {
                    penalty = Double.parseDouble(args[count+1]);
                    if(penalty < 0)
                    {
                        System.err.println("Invalid value for penalty: " + penalty);
                        System.exit(-1);
                    }
                    count+=2;
                }
                else if(args[count].equals("-rv"))
                {
                    toRemove = new ArrayList<String>();
                    int index = count+1;
                    while(index < args.length && !args[index].startsWith("-"))
                    {
                        toRemove.add(args[index]);
                        index++;
                    }
                    count = index;
                }
                else if(args[count].equals("-runMGM"))
                {
                    runMGM = true;
                    count++;
                }
                else
                {
                    throw new Exception("Unsupported Command Line Switch: " + args[count]);
                }
            }
            //Exit program if no data file is specified
            if(dataFile.equals(""))
            {
                System.err.println("Usage: java -jar causalDiscovery.jar -d <Data File>\n Please specify a data file");
                System.exit(-1);
            }
            //Load specified dataset from the file
            d = MixedUtils.loadDataSet2(dataFile,maxNumDiscrete);
            System.out.println("Loaded dataset: " + d + "\n Dataset is mixed? " + d.isMixed());


            for(String s: toRemove)
            {
                d.removeColumn(d.getVariable(s));
            }
            //Exit program if no algorithm is specfiied
            if(alg.equals("None")&&!runMGM &&!runSteps && !runsurvivalMGM)
            {
                System.out.println("No algorithm specified, and you are not running MGM\nTherefore no method will be run to analyze the data.\n Please use -runMGM to run MGM or -a <Algorithm Name> to run a causal discovery method");
                System.exit(-1);
            }
            String [] algos = {"FCI","FCI50","CFCI","FCI-MAX","PCS","CPC","PC50","MAX","FGES","LiNG", "None"};
            boolean foundAl = false;
            for(String x:algos)
            {
                if(x.equals(alg))
                    foundAl = true;
            }

            //Ensure that algorithm name is valid
            if(!foundAl)
                throw new Exception("Unknown Algorithm: " + alg + ", Please use either \"FCI\", \"FCI50\", \"CFCI\", \"FCI-MAX\", \"PCS\", \"CPC\", \"PC50\", \"MAX\", \"FGES\", \"LiNG\" or \"None\" (MGM Only)");


            //Workflow for running StEPS to compute optimal Lambda parameters
            if(runSteps) {

                if (!d.isMixed())
                {
                    System.err.println("Cannot run StEPS on a dataset that isn't mixed discrete and continuous...exiting");
                    System.exit(-1);
                }


                //Compute range of lambda parameters based on user specified input
                double low = .05;
                double high = .9;
                if(paramLow!=-1)
                    low = paramLow;
                if(paramHigh!=-1)
                    high = paramHigh;
                double[] initLambdas;
                if (paramMult==-1) {
                    initLambdas = new double[numParams];
                    for (int i = 0; i < numParams; i++) {
                        initLambdas[i] = i * (high - low) / numParams + low;
                    }
                    System.out.println("Running StEPS with lambda range (" + low + "," + high + "), testing " + numParams + " params...");
                } else {
                    int count = 0;
                    double temp = low;
                    while (temp < high) {
                        temp *= paramMult;
                        count++;
                    }
                    initLambdas = new double[count];
                    initLambdas[0] = low;
                    for (int i = 1; i < count; i++) {
                        initLambdas[i] = paramMult * initLambdas[i-1];
                    }
                    System.out.print("Running StEPS with lambdas (");
                    for (int i = 0; i < count-1; i++) {
                        System.out.print(Math.round(1000 * initLambdas[i]) / 1000.0 + ", ");
                    }
                    System.out.println(Math.round(1000 * initLambdas[count-1]) / 1000.0 + ")...");
                }
                STEPS s;


                //If size of subsamples are not given, these will be computed automatically
                if(b==-1)
                    s = new STEPS(d.copy(),initLambdas,threshold,ns);
                else
                    s = new STEPS(d.copy(),initLambdas,threshold,ns,b);
                s.setComputeStabs(outputStabs);


                //Run 
                s.runStepsPar();

                //Get stabilities and optimal lambda
                double [][] stabs = s.stabilities;
                double [] lbm = s.lastLambda;



                //Output edge appearence stability to file
                if(outputStabs)
                {
                    PrintStream stabOut = new PrintStream(stabPath);
                    edu.pitt.csb.mgm.runSteps.printStability(stabOut,d,stabs);
                }
                if (runMGM) {
                	lambda = s.lastLambda;
                	PrintStream out = new PrintStream(outputPath);
                	out.println(lbm[0] + "\t" + lbm[1] + "\t" + lbm[2]);
                	out.flush();
                    out.close();
                }
                if (runsurvivalMGM) {
                	survlambda = s.lastLambda;
                	PrintStream out = new PrintStream(outputPath);
                	out.println(lbm[0] + "\t" + lbm[1] + "\t" + lbm[2]);
                	out.flush();
                    out.close();
                }
                System.out.println("Done");
            }

            Graph g = null;


            //Workflow to run MGM based on optimal lambda or user specified lambda
            if(runMGM) {
                if(!d.isMixed())
                {
                    System.out.println("Dataset is not mixed continuous and discrete... cannot run MGM");
                    System.exit(-1);
                }
                if(runCPSS) //MGM with CPSS
                {
                    Bootstrap bs = new Bootstrap(d,cpssBound,50);
                    bs.setCPSS();
                    g = bs.runBootstrap(convert("MGM"),lambda);
                    if(outputOrients)
                        outputBootOrients(bs.getTabularOutput());
                }
                else if(runBootstrap) //MGM with Bootstrapping
                {
                    Bootstrap bs = new Bootstrap(d,bootBound,numBoots);
                    g = bs.runBootstrap(convert("MGM"),lambda);
                    if(outputOrients)
                        outputBootOrients(bs.getTabularOutput());
                }
                else //Vanilla MGM
                {
                    System.out.print("Running MGM with lambda params: " + Arrays.toString(lambda) + "...");
                    MGM m = new MGM(d, lambda); //Create MGM object
                    m.learn(1e-5, 1000);//Use maximum 1000 iterations to learn the edges for the undirected MGM graph, stop searching if the edges in the graph don't change after 3 iterations
                    g = m.graphFromMGM(); //store the mgm graph
                    System.out.println("Done");
                    if(outputOrients)
                        System.out.println("For subsampled edge frequency information for MGM, please run StEPS");
                }
            }
            if(runsurvivalMGM) {
                if(!d.isMixed())
                {
                    System.out.println("Dataset is not mixed continuous and discrete... cannot run MGM");
                    System.exit(-1);
                }
                if(runCPSS) //MGM with CPSS
                {
                    Bootstrap bs = new Bootstrap(d,cpssBound,50);
                    bs.setCPSS();
                    g = bs.runBootstrap(convert("MGM"),survlambda);
                    if(outputOrients)
                        outputBootOrients(bs.getTabularOutput());
                }
                else if(runBootstrap) //MGM with Bootstrapping
                {
                    Bootstrap bs = new Bootstrap(d,bootBound,numBoots);
                    g = bs.runBootstrap(convert("MGM"),survlambda);
                    if(outputOrients)
                        outputBootOrients(bs.getTabularOutput());
                }
                else if (solvePath) {
                    //Compute range of lambda parameters based on user specified input
                    double low = .05;
                    double high = .9;

                    if(paramLow!=-1)
                        low = paramLow;
                    if(paramHigh!=-1)
                        high = paramHigh;

                    double[] initLambdas;
                    if (paramMult==-1) {
                        initLambdas = new double[numParams];
                        for (int i = 0; i < numParams; i++) {
                            initLambdas[i] = i * (high - low) / numParams + low;
                        }
                        System.out.println("Solution path with lambda range (" + low + "," + high + "), using " + numParams + " params...");
                    } else {
                        int count = 0;
                        double temp = low;
                        while (temp < high) {
                            temp *= paramMult;
                            count++;
                        }
                        initLambdas = new double[count];
                        initLambdas[0] = low;
                        for (int i = 1; i < count; i++) {
                            initLambdas[i] = paramMult * initLambdas[i-1];
                        }
                        System.out.print("Solution path with lambdas (");
                        for (int i = 0; i < count-1; i++) {
                            System.out.print(Math.round(1000 * initLambdas[i]) / 1000.0 + ", ");
                        }
                        System.out.println(Math.round(1000 * initLambdas[count-1]) / 1000.0 + ")...");
                    }

                    for (int i = 0; i < 5; i++) survlambda[i] = initLambdas[initLambdas.length-1];

                    survivalMGM m = new survivalMGM(d, survlambda);
                    ArrayList<survivalMGM.survivalMGMParams> path = m.learnPathEdges(500, initLambdas);

                    for (int i = 0; i < initLambdas.length; i++) {
                        m.setParams(path.get(i));
                        g = m.graphFromMGM();
                        if(outputSif)
                        {
                            String tempPath = sifPath.substring(0, sifPath.indexOf(".sif")) + "_l" + initLambdas[i] + ".sif";
                            PrintStream out = new PrintStream(tempPath);
                            printSif(g,out);
                        }
                        String tempPath = outputPath.substring(0, outputPath.indexOf(".txt")) + "_l" + initLambdas[i] + ".txt";
                        PrintStream out = new PrintStream(tempPath);
                        out.println(g);
                        out.flush();
                        out.close();
                    }
                    System.out.println("Done");
                }
                else //Vanilla survivalMGM
                {
                    System.out.print("Running survivalMGM with lambda params: " + Arrays.toString(survlambda) + "...");
                    survivalMGM m = new survivalMGM(d, survlambda); //Create MGM object
                    m.learnEdges(1000);//Use maximum 1000 iterations to learn the edges for the undirected MGM graph, stop searching if the edges in the graph don't change after 3 iterations
                    g = m.graphFromMGM(); //store the mgm graph
//                    m.moralizeCensoredNeighbors(d, g, alpha);
                    System.out.println("Done");
                    if(outputOrients)
                        System.out.println("For subsampled edge frequency information for MGM, please run StEPS");
                }
            }
            Graph finalOutput = null;
            IKnowledge k = null;
            //Incorporate prior knowledge file into cd algorithms if specified
            if(useKnowledge) {
                if (kFile == null) {
                    throw new IllegalStateException("No knowledge data file was specified.");
                }

                try {
                    File knowledgeFile = new File(kFile);

                    CharArrayWriter writer = new CharArrayWriter();

                    FileReader fr = new FileReader(knowledgeFile);
                    int i;

                    while ((i = fr.read()) != -1) {
                        writer.append((char) i);
                    }

                    DataReader reader = new DataReader();
                    char[] chars = writer.toCharArray();

                    k = reader.parseKnowledge(chars);
                }
                catch(Exception e)
                {
                    e.printStackTrace();
                    System.out.println("Unable to read knowledge file");
                    return;
                }
            }

            //Run StARS to compute optimal parameter for specifieid cd algorithm
            if(runStars)
            {
                if(alg.equals("None"))
                {
                    System.err.println("Cannot run StARS without a specified directed causal discovery algorithm");
                    System.exit(-1);
                }
                double pLow = 0.0001;
                double pHigh = 0.8;

                //Tailor parameter range to algorithm being used
                if(alg.equals("FGES") && (d.isMixed() || d.isDiscrete()))
                {
                    pLow = 1;
                    pHigh = 10;
                }
                else if(alg.equals("FGES") && d.isContinuous())
                {
                    pLow = 0.01;
                    pHigh = 20;
                }

                //Or use user specified paramter range
                if(paramLowStars!=-1)
                    pLow = paramLowStars;
                if(paramHighStars!=-1)
                    pHigh = paramHighStars;

                //Compute paramter range
                double [] penalties = new double[numParams];
                for(int j = 0; j < penalties.length;j++)
                {
                    penalties[j] = pLow + j*(pHigh-pLow)/numParams;
                }


                STARS strs;
                Algorithm a = convert(alg);

                System.out.print("Running StARS for algorithm, " + a + " with " + numParams + " parameters in the range (" + pLow + "," + pHigh + ")...");
                if(b==-1)
                    strs = new STARS(d,penalties,threshold,ns,a);
                else
                    strs = new STARS(d,penalties,threshold,ns,a,b);
                //Run StARS to get optimal parameter, need to flip parameter range if FGS is being run (low to high)
                penalty = strs.getAlpha(a==Algorithm.FGS);
                System.out.println("STARS Chosen Parameter: " + penalty);


                //Output edge appearence stabilities (this will overwrite StEPS stabilities)
                double [][] stab = strs.stabilities; //This is edge stability
                if(outputStabs)
                {
                    PrintStream stabOut = new PrintStream(stabPath);
                    edu.pitt.csb.mgm.runSteps.printStability(stabOut,d,stab);
                }
            }

            //Run Regular CD Algorithm
            if(!runCPSS && !runBootstrap) {

                //Run any algorithm that isn't FGES
                if(!alg.equals("None") && !alg.equals("FGES")) {
                    System.out.println("Running " + alg + "...");
                    System.out.println("Maximum Heap Space: " + Runtime.getRuntime().maxMemory());
                    DataGraphSearch gs = Algorithm.algToSearchWrapper(convert(alg), new double[]{alpha});
                    if(g!=null) {
                        if (d.isCensored()) {
                            g = moralizeCensoredNeighbors(d, g, alpha);
                        }
                        gs.setInitialGraph(g);
                    } else if(initGraph!=null) {
                        if (d.isCensored()) {
                            initGraph = moralizeCensoredNeighbors(d, initGraph, alpha);
                        }
                        gs.setInitialGraph(initGraph);
                    }
                    if(k!=null)
                        gs.setKnowledge(k);
                    finalOutput = gs.search(d);
                    System.out.println("Done");
                }
                //Run FGES
                else if(alg.equals("FGES"))
                {
                    try{
                        System.out.print("Running FGES...");
                        Score s;
                        if(d.isMixed()) {
                            ConditionalGaussianScore s2 = new ConditionalGaussianScore(d);
                            s2.setStructurePrior(penalty);
                            s = s2;
                        }
                        else if(d.isContinuous()) {
                            s = new SemBicScore(new CovarianceMatrixOnTheFly(d), penalty);
                        }
                        else {
                            BDeuScore s2 = new BDeuScore(d);
                            s2.setStructurePrior(penalty);
                            s = s2;
                        }
                            Fges fg = new Fges(s);
                            if (useKnowledge)
                                fg.setKnowledge(k);
                            if(g!=null)
                                fg.setInitialGraph(g);
                            else if(initGraph!=null)
                                fg.setInitialGraph(initGraph);
                            finalOutput = fg.search();

                        System.out.println("Done");
                    }
                    catch(Exception e)
                    {
                        System.err.println("Error Running FGES, check for collinearity");
                        e.printStackTrace();
                        System.exit(-1);
                    }

                    System.out.println("Done");
                }

                //Otherwise just save MGM output as the final output
                else
                {
                    finalOutput = g;
                }


                //This will output orientation stabilities for all edges that showed up at least once with some orientation
                //Could consider changing this to edges that showed up in the output graph only, but this will require a bit of post-processing
                //TODO
                if(outputOrients)
                {
                    Bootstrap bs = new Bootstrap(d,1,ns);
                    if(b<=0)
                        bs.setSubsample(0);
                    else
                        bs.setSubsample(b);
                    if(alg.equals("None"))
                    {
                        System.out.println("Cannot output orientation stability without a specified causal discovery algorithm");
                    }
                    else {
                        double param = 0;
                        if (alg.equals("FGES"))
                            param = penalty;
                        else
                            param = alpha;
                        bs.runBootstrap(convert(alg),new double[]{param});
                        outputBootOrients(bs.getTabularOutput());
                    }
                }
            }
            //Complimentary Pairs Stability Selection for all algorithms
            else if(runCPSS)
            {
                Bootstrap bs = new Bootstrap(d, cpssBound, 50);
                bs.setCPSS();
                if(!alg.equals("FGES") && !alg.equals("None")) {

                    finalOutput = bs.runBootstrap(convert(alg), new double[]{alpha});
                }
                else if(!alg.equals("None")) //Use penalty instead of alpha for FGES
                {
                    finalOutput = bs.runBootstrap(convert(alg), new double[]{penalty});
                }
                if(outputOrients)
                    outputBootOrients(bs.getTabularOutput());

            }
            //Bootstrapping, same workflow as CPSS (but no bs.setCPSS() line)
            else if(runBootstrap)
            {
                Bootstrap bs = new Bootstrap(d,bootBound,numBoots);
                if(!alg.equals("FGES") && !alg.equals("None")) {

                    finalOutput = bs.runBootstrap(convert(alg),new double[]{alpha});
                }
                else if(!alg.equals("None"))
                {
                    finalOutput = bs.runBootstrap(convert(alg),new double[]{penalty});
                }
                if(outputOrients)
                    outputBootOrients(bs.getTabularOutput());
            }

            if(outputSif)
            {
                PrintStream out = new PrintStream(sifPath);
                printSif(finalOutput,out);
            }
            PrintStream out = new PrintStream(outputPath);
            out.println(finalOutput);
            out.flush();
            out.close();




        } catch (IOException e){
            e.printStackTrace();
        }
    }


    public static void outputBootOrients(String o) throws Exception
    {
        PrintStream out = new PrintStream(orientPath);
        out.println(o);
        out.flush();
        out.close();
    }
    public static void printSif(Graph g, PrintStream out)
    {
        for(Edge e:g.getEdges()) {
            Endpoint ep1 = e.getEndpoint1();
            Endpoint ep2 = e.getEndpoint2();
            if (ep1 == Endpoint.TAIL) {
                if (ep2 == Endpoint.TAIL) {
                    out.println(e.getNode1() + "\tundir\t" + e.getNode2());
                } else if (ep2 == Endpoint.ARROW) {
                    out.println(e.getNode1() + "\tdir\t" + e.getNode2());
                }

            } else if (ep1 == Endpoint.CIRCLE) {
                if (ep2 == Endpoint.CIRCLE)
                    out.println(e.getNode1() + "\tcc\t" + e.getNode2());
                else if (ep2 == Endpoint.ARROW)
                    out.println(e.getNode1() + "\tca\t" + e.getNode2());
            } else {
                if (ep2 == Endpoint.TAIL)
                    out.println(e.getNode2() + "\tdir\t" + e.getNode1());
                else if(ep2==Endpoint.CIRCLE)
                    out.println(e.getNode2() + "\tca\t" + e.getNode1());
                else
                    out.println(e.getNode1() + "\tbidir\t" + e.getNode2());
            }
        }
        out.flush();
        out.close();
    }


    private static Graph moralizeCensoredNeighbors(DataSet ds, Graph g, double alpha) throws RuntimeException {
        IndependenceTest test = new IndTestMultinomialAJ(ds, alpha, true);
        List<Node> variables = new ArrayList<>();
        List<Node> neighbors = new ArrayList<>();
        List<Node> zList = new ArrayList<>();
        List<Node> source = new ArrayList<>();
        List<Node> target = new ArrayList<>();

        DataSet dsCont = MixedUtils.getContinousData(ds);
        DataSet dsDisc = MixedUtils.getDiscreteData(ds);
        DataSet dsSurv = MixedUtils.getCensoredData(ds);

        variables.addAll(dsCont.getVariables());
        variables.addAll(dsDisc.getVariables());
        variables.addAll(dsSurv.getVariables());

//        int p = dsCont.getNumColumns();
//        int q = dsDisc.getNumColumns();
        int r = dsSurv.getNumColumns();

//        for (Node n : variables) {
//            if (n instanceof ContinuousVariable) p++;
//            else if (n instanceof DiscreteVariable) q++;
//            else if (n instanceof CensoredVariable) r++;
//            else throw new RuntimeException("Node " + n.getName() + " has an unknown variable type");
//        }

        int[] censIdx = new int[r];

        int count = 0;
        for (int i = 0; i < variables.size(); i++) {
            if (variables.get(i) instanceof CensoredVariable) {
                censIdx[count] = i;
                count++;
                if (count == r) break;
            }
        }

        System.out.println("Moralizing graph with " + r + " censored variables");

        for (int idx : censIdx) {
            neighbors.clear();

            for (Node n : variables) {
                if (g.isAdjacentTo(g.getNode(n.getName()), g.getNode(variables.get(idx).getName()))) neighbors.add(n);
            }

            if (neighbors.isEmpty()) continue;

//            System.out.println(variables.get(idx).getName() + ":\t" + neighbors.size() + " neighbors");

            boolean skip;
            for (int i = 0; i < neighbors.size()-1; i++) {
                for (int k = i+1; k < neighbors.size(); k++) {
                    if (!g.isAdjacentTo(g.getNode(neighbors.get(i).getName()), g.getNode(neighbors.get(k).getName()))) {
//                        System.out.println(variables.get(idx).getName() + ":\t" + neighbors.get(i) + ", " + neighbors.get(k) + " not adjacent");
                        skip = false;
                        for (int m = 0; m < source.size(); m++) {
                            if ((source.get(m) == neighbors.get(i) && target.get(m) == neighbors.get(k)) ||
                                    (source.get(m) == neighbors.get(k) && target.get(m) == neighbors.get(i))) {
                                skip = true;
                                break;
                            }
                        }
                        if (skip) continue;
                        zList.clear();
                        zList.add(variables.get(idx));
                        for (Node n : variables) {
                            if (g.isAdjacentTo(g.getNode(n.getName()), g.getNode(neighbors.get(i).getName()))
                                    || g.isAdjacentTo(g.getNode(n.getName()), g.getNode(neighbors.get(k).getName()))) {
                                if (!(zList.contains(n))) {
                                    zList.add(n);
                                }
                            }
                        }
//                        System.out.println(variables.get(idx).getName() + ":\tzList " + zList);
                        if (test.isDependent(neighbors.get(i), neighbors.get(k), zList)) {
//                            System.out.println(variables.get(idx).getName() + ":\t" + neighbors.get(i) + ", " + neighbors.get(k) + " are dependent");
                            System.out.println("Edge added between " + neighbors.get(i) + " and " + neighbors.get(k));

                            source.add(neighbors.get(i));
                            target.add(neighbors.get(k));
//                            g.addUndirectedEdge(neighbors.get(i), neighbors.get(k));
                        }
                    }
                }
            }
        }

        for (int i = 0; i < source.size(); i++) g.addUndirectedEdge(g.getNode(source.get(i).getName()), g.getNode(target.get(i).getName()));

//        System.out.println(g);

        return g;
    }




    //Helper method to convert a string to an Algorithm representation
    private static Algorithm convert(String alg)
    {
        Algorithm a;
        if(alg.equals("CPC"))
        {
            a = Algorithm.CPC;
        }
        else if(alg.equals("PC50"))
        {
            a = Algorithm.PC50;
        }
        else if(alg.equals("MAX"))
        {
            a = Algorithm.PCMAX;
        }
        else if(alg.equals("FCI"))
        {
            a = Algorithm.FCI;
        }
        else if(alg.equals("FCI50"))
        {
            a = Algorithm.FCI50;
        }
        else if(alg.equals("CFCI"))
        {
            a = Algorithm.CFCI;
        }
        else if(alg.equals("FCI-MAX"))
        {
            a = Algorithm.FCIMAX;
        }
        else if(alg.equals("FGES"))
        {
            a = Algorithm.FGS;
        }
        else if(alg.equals("LiNG"))
        {
            a = Algorithm.LiNG;
        }
        else if(alg.equals("None"))
        {
            a = Algorithm.MGM;
        }
        else
        {
            a = Algorithm.PCS;
        }
        return a;
    }
}

