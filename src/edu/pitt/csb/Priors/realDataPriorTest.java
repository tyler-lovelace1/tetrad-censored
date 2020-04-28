package edu.pitt.csb.Priors;

import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.util.TetradMatrix;
import edu.pitt.csb.mgm.MixedUtils;
import edu.pitt.csb.mgm.STEPS;
import edu.pitt.csb.stability.CrossValidationSets;
import edu.pitt.csb.stability.StabilityUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by vinee_000 on 12/6/2017.
 */
public class realDataPriorTest {

    public static void main(String [] args) throws Exception
    {

        //Three Groups, No Prior = run with the data alone (just to compute the PAM50 enrichment and prediction accuracy)
        //Irrelevant Prior = run with the Irrelevant PAM50 Genesets alone (also to compute PAM50 enrichment and prediction accuracy)
        //Relevant Prior = run with Irrelevant and actual PAM50 to compare weighting scores, and to compute Prediction accuracy of this network
        //Pathway Prior = run with just all of the pathways to compute weighting


        boolean runNoPrior = false; //No Priors at all for PAM50
        boolean runIrrelevantPrior = false; //Only the Irrelevant PAM50 Priors
        boolean runRelevantPrior = false; //Both Relevant PAM50 and Irrelevant PAM50
        boolean doNumPriors = false; //Use a different number of priors, as specified by the below parameter
        boolean runOnlyRelevant = false; //Only the Relevant PAM50 Prior
        boolean runBoth = true; //Pathway Priors
        boolean tumors = false; //tumor or normal samples?
        boolean erPositive = false;
        boolean erNegative = false;
        int type = -1;
       // boolean computeMetrics = false;
        boolean useStabilities = false;

        int numLambda = 40;
        int numPriors = 5;
        double g = 0.01;

        double low = .05;
        double high = .9;
        int k = 10;

        String [] types = {"Luminal_A", "Luminal_B","Triple_Negative", "HER2"};

        System.out.print("Parsing Arguments...");

        int index = 0;
        while(index < args.length)
        {
            if(args[index].equals("-np"))
            {
                runNoPrior = true;
                index++;
            }
            else if(args[index].equals("-numPrior"))
            {
                numPriors = Integer.parseInt(args[index+1]);
                doNumPriors = true;
                index+=2;
            }
            else if(args[index].equals("-ip"))
            {
                runIrrelevantPrior = true;
                index++;
            }
            else if(args[index].equals("-rp"))
            {
                runRelevantPrior = true;
                index++;
            }
            else if(args[index].equals("-pp"))
            {
                runBoth = true;
                index++;
            }
            else if(args[index].equals("-t"))
            {
                tumors = true;
                index++;
            }
            else if(args[index].equals("-nl"))
            {
                numLambda = Integer.parseInt(args[index+1]);
                index+=2;
            }
            else if(args[index].equals("-g"))
            {
                g = Double.parseDouble(args[index+1]);
                index+=2;
            }
            else if (args[index].equals("-us"))
            {
                useStabilities = true;
                index++;
            }
            else if(args[index].equals("-rorp"))
            {
                runOnlyRelevant=true;
                index++;
            }
            else if(args[index].equals("-er+"))
            {
                erPositive = true;
                index++;
            }
            else if(args[index].equals("-er-"))
            {
                erNegative = true;
                index++;
            }
            else if(args[index].equals("-type"))
            {
                type = Integer.parseInt(args[index + 1]);
                index+=2;
            }
            else if(args[index].equals("-kfold"))
            {
                k = Integer.parseInt(args[index+1]);
                index+=2;
            }

        }

        System.out.println("Done");

        double [] lambda = new double[numLambda];
        double inc = (high-low)/(numLambda-1);
        for(int i = 0; i < numLambda;i++)
        {
            lambda[i] = i*inc + low;
        }
        DataSet data = null;


        System.out.print("Reading Subsamples...");
        data = MixedUtils.loadDataSet2("genes_with_clinical.txt");
        if(erPositive)
            data = MixedUtils.loadDataSet2("ER_Positive.txt");
        else if(erNegative)
            data = MixedUtils.loadDataSet2("ER_Negative.txt");
        else if(type!=-1)
            data = MixedUtils.loadDataSet2(types[type] + ".txt");
        else if(!tumors)
            data = MixedUtils.loadDataSet2("genes_with_clinical_normals.txt");
        //System.out.println(data);
        System.out.println(data);
        data = MixedUtils.completeCases(data);

        //System.out.println(data);
        File f = new File("Subsamples");
        if(erPositive)
            f = new File("ER_Positive_Subsamples");
        else if(erNegative)
            f = new File("ER_Negative_Subsamples");
        else if(!tumors)
            f = new File("Normal_Subsamples");
        else if(type!=-1)
            f = new File(types[type] + "_Subsamples");
        if(!f.exists())
            f.mkdir();
        int numSub = 20;

        int b = (int) Math.floor(10 * Math.sqrt(data.getNumRows()));
        if (b > data.getNumRows())
            b = data.getNumRows() / 2;
        int[][] samps = StabilityUtils.subSampleNoReplacement(data.getNumRows(), b, numSub);
            File temp = new File("Subsamples/Subsamples.txt");
            if(erPositive)
                temp = new File("ER_Positive_Subsamples/ER_Positive_Subsamples.txt");
            else if(erNegative)
                temp = new File("ER_Negative_Subsamples/ER_Negative_Subsamples.txt");
            else if(!tumors)
                temp = new File("Normal_Subsamples/Normal_Subsamples.txt");
            else if(type!=-1)
                temp = new File(types[type] + "_Subsamples/" + types[type] + "_Subsamples.txt");
            if(!temp.exists())
            {
                PrintStream out2 = new PrintStream(temp.getAbsolutePath());
                for(int i = 0; i < samps.length;i++) {

                    for (int j = 0; j < samps[i].length; j++) {
                        if(j==samps[i].length-1)
                            out2.println(samps[i][j]);
                        else
                            out2.print(samps[i][j] + "\t");
                    }
                }
                out2.flush();
                out2.close();
            }
            else
            {
                BufferedReader b2 = new BufferedReader(new FileReader(temp));
                for(int j = 0; j < numSub;j++)
                {
                    String [] line = b2.readLine().split("\t");
                    samps[j] = new int[line.length];
                    for(int l = 0; l< line.length;l++)
                        samps[j][l] = Integer.parseInt(line[l]);
                }
            }

        System.out.println("Done");
        //Test no Prior situation
        if(runNoPrior) {
            STEPS s = new STEPS(data, lambda, g, samps);
            Graph graph = s.runStepsPar();
            double[][] stab = s.stabilities;
            //TODO Should parallelize this, but want to be safe because of the time crunch for the paper deadline
            System.out.println("Cross Validating");
            CrossValidationSets cv = new CrossValidationSets(data,s.lastLambda,"Subsamples",".","Subtype",k,"No_Prior");
            cv.crossValidate();
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            PrintStream out;
            if(tumors)
                out = new PrintStream("Stabilities_No_Prior.txt");
            else
                out = new PrintStream("Stabilities_No_Prior_Normals.txt");
            for (int i = 0; i < data.getNumColumns(); i++) {
                if (i == data.getNumColumns() - 1)
                    out.println(data.getVariable(i).getName());
                else
                    out.print(data.getVariable(i).getName() + "\t");
            }
            for (int i = 0; i < stab.length; i++) {
                out.print(data.getVariable(i).getName() + "\t");
                for (int j = 0; j < stab[i].length; j++) {
                    if (j == stab[i].length - 1)
                        out.println(stab[i][j]);
                    else
                        out.print(stab[i][j] + "\t");
                }
            }
            out.flush();
            out.close();
            if(tumors)
            out = new PrintStream("Graph_No_Prior.txt");
            else
                out = new PrintStream("Graph_No_Prior_Normals.txt");
            out.println(graph);
            out.flush();
            out.close();
        }

        //Test Irrelevant Prior Situation

        if(runIrrelevantPrior)
        {

            numPriors = 5;
            SparseDoubleMatrix2D[] priors = new SparseDoubleMatrix2D[numPriors];
            for(int i = 0; i < numPriors;i++) {
                //priors[i] = new TetradMatrix(loadPrior(new File("prior_sources/Irr_Prior_" + i + ".txt"),data.getNumColumns()));
                priors[i] = new SparseDoubleMatrix2D(loadPAM50(new File("prior_sources/Irr_PAM50_" + i + ".txt"),data.getNumColumns()));
                System.out.println(priors[i].rows() + "," + priors[i].columns());
            }
            System.out.println("Constructing lambdas...");
            mgmPriors m = new mgmPriors(samps.length,lambda,data,priors,samps);
            PrintStream lb = new PrintStream("Irrelevant_Lambdas.txt");
            for(int i = 0; i < m.getLambdas().length;i++)
            {
                lb.println(Arrays.toString(m.getLambdas()[i]));
            }
            lb.flush();
            lb.close();
            System.out.println("Running Priors...");
            Graph g2 = m.runPriors();

            //TODO SAME AS ABOVE SHOULD BE PARALLELIZED BUT TIME CRUNCH
            System.out.println("Cross Validating");
            CrossValidationSets cv = new CrossValidationSets(data,m.lastNPLambda,m.lastWPLambda,"Subsamples",".","Subtype",m.lastHavePrior,k,"Irrelevant_Prior");
            cv.crossValidate();
            //////////////////////////////////////////////////////////////


            /////////////////////
            double [][] stab = m.edgeScores;
            double [] weights = m.normalizedExpertWeights;
            double [] pValues = m.pValues;

            PrintStream w;
            if(tumors)
                w = new PrintStream("Weights_Irrelevant_Prior.txt");
            else
                w = new PrintStream("Weights_Irrelevant_Prior_Normals.txt");
            for(int i = 0; i < weights.length;i++)
            {
                w.println("Irr_PAM50_" + i  + "\t" + weights[i] + "\t" + pValues[i]);
            }
            w.flush();
            w.close();
            PrintStream out;
            if(tumors)
                out = new PrintStream("Stabilities_Irrelevant_Prior.txt");
            else
                out = new PrintStream("Stabilities_Irrelevant_Prior_Normals.txt");
            for (int i = 0; i < data.getNumColumns(); i++) {
                if (i == data.getNumColumns() - 1)
                    out.println(data.getVariable(i).getName());
                else
                    out.print(data.getVariable(i).getName() + "\t");
            }
            for (int i = 0; i < stab.length; i++) {
                out.print(data.getVariable(i).getName() + "\t");
                for (int j = 0; j < stab[i].length; j++) {
                    if (j == stab[i].length - 1)
                        out.println(stab[i][j]);
                    else
                        out.print(stab[i][j] + "\t");
                }
            }
            out.flush();
            out.close();
            if(tumors)
            out = new PrintStream("Graph_Irrelevant_Prior.txt");
            else
                out = new PrintStream("Graph_Irrelevant_Prior_Normals.txt");
            out.println(g2);
            out.flush();
            out.close();


        }


        if(runOnlyRelevant)
        {
            SparseDoubleMatrix2D[] priors = new SparseDoubleMatrix2D[1];
            priors[0] = new SparseDoubleMatrix2D(loadPAM50(new File("prior_sources/Prior_PAM50.txt"),data.getNumColumns()));

            mgmPriors m = new mgmPriors(samps.length,lambda,data,priors,samps);
            PrintStream lb;
            if(tumors)
                lb = new PrintStream("Only_Relevant_Lambdas.txt");
            else
                lb = new PrintStream("Only_Relevant_Lambdas_Normals.txt");
            for(int i = 0; i < m.getLambdas().length;i++)
            {
                lb.println(Arrays.toString(m.getLambdas()[i]));
            }
            lb.flush();
            lb.close();
            System.out.println("Running Priors...");
            Graph g2 = m.runPriors();


            //TODO SAME AS ABOVE SHOULD BE PARALLELIZED BUT TIME CRUNCH
            System.out.println("Cross Validating...");
            CrossValidationSets cv = new CrossValidationSets(data,m.lastNPLambda,m.lastWPLambda,"Subsamples",".","Subtype",m.lastHavePrior,k,"Only_Relevant_Prior");
            cv.crossValidate();
            System.out.println("Done");
            //////////////////////////////////////////////////////////////
            double [][] stab = m.edgeScores;
            double [] weights = m.normalizedExpertWeights;
            double [] pValues = m.pValues;


            PrintStream w;
            if(tumors)
                w = new PrintStream("Weights_Only_Relevant_Prior.txt");
            else
                w = new PrintStream("Weights_Only_Relevant_Prior_Normals.txt");
            for(int i = 0; i < weights.length;i++)
            {
                if(i==weights.length-1)
                    w.println("PAM50\t" + weights[i] + "\t" + pValues[i]);
                else
                    w.println("Irr_PAM50_" + i + "\t" + weights[i] + "\t" + pValues[i]);
            }
            w.flush();
            w.close();
            PrintStream out;
            if(tumors)
                out = new PrintStream("Stabilities_Only_Relevant_Prior.txt");
            else
                out = new PrintStream("Stabilities_Only_Relevant_Prior_Normals.txt");
            for (int i = 0; i < data.getNumColumns(); i++) {
                if (i == data.getNumColumns() - 1)
                    out.println(data.getVariable(i).getName());
                else
                    out.print(data.getVariable(i).getName() + "\t");
            }
            for (int i = 0; i < stab.length; i++) {
                out.print(data.getVariable(i).getName() + "\t");
                for (int j = 0; j < stab[i].length; j++) {
                    if (j == stab[i].length - 1)
                        out.println(stab[i][j]);
                    else
                        out.print(stab[i][j] + "\t");
                }
            }
            out.flush();
            out.close();
            if(tumors)
                out = new PrintStream("Graph_Only_Relevant_Prior.txt");
            else
                out = new PrintStream("Graph_Only_Relevant_Prior_Normals.txt");
            out.println(g2);
            out.flush();
            out.close();

        }


        //Test Relevant PAM 50 Situation
        if(runRelevantPrior)
        {

            if(!doNumPriors)
                numPriors = 5;
            SparseDoubleMatrix2D [] priors = new SparseDoubleMatrix2D[numPriors+1];
            for(int i = 0; i < numPriors;i++) {
                priors[i] = new SparseDoubleMatrix2D(loadPAM50(new File("prior_sources/Irr_PAM50_" + i + ".txt"),data.getNumColumns()));

            }
            priors[priors.length-1] = new SparseDoubleMatrix2D(loadPAM50(new File("prior_sources/Prior_PAM50.txt"),data.getNumColumns()));

            mgmPriors m = new mgmPriors(samps.length,lambda,data,priors,samps);
            PrintStream lb;
            if(tumors)
                lb = new PrintStream("Relevant_Lambdas.txt");
            else
                lb = new PrintStream("Relevant_Lambdas_Normals.txt");
            for(int i = 0; i < m.getLambdas().length;i++)
            {
                lb.println(Arrays.toString(m.getLambdas()[i]));
            }
            lb.flush();
            lb.close();
            System.out.println("Running Priors...");
            Graph g2 = m.runPriors();


            //TODO SAME AS ABOVE SHOULD BE PARALLELIZED BUT TIME CRUNCH
            System.out.println("Cross Validating...");
            String runName = "Relevant_Priors";
            if(doNumPriors)
                runName ="Relevant_Prior_" + numPriors;
            CrossValidationSets cv = new CrossValidationSets(data,m.lastNPLambda,m.lastWPLambda,"Subsamples",".","Subtype",m.lastHavePrior,k,runName);
            cv.crossValidate();
            System.out.println("Done");
            //////////////////////////////////////////////////////////////
            double [][] stab = m.edgeScores;
            double [] weights = m.normalizedExpertWeights;
            double [] pValues = m.pValues;


            PrintStream w;
            if(tumors)
                 w = new PrintStream("Weights_Relevant_Prior.txt");
            else
                w = new PrintStream("Weights_Relevant_Prior_Normals.txt");
            for(int i = 0; i < weights.length;i++)
            {
                if(i==weights.length-1)
                    w.println("PAM50\t" + weights[i] + "\t" + pValues[i]);
                else
                    w.println("Irr_PAM50_" + i + "\t" + weights[i] + "\t" + pValues[i]);
            }
            w.flush();
            w.close();
            PrintStream out;
            if(tumors)
                out = new PrintStream("Stabilities_Relevant_Prior.txt");
            else
                 out = new PrintStream("Stabilities_Relevant_Prior_Normals.txt");
            for (int i = 0; i < data.getNumColumns(); i++) {
                if (i == data.getNumColumns() - 1)
                    out.println(data.getVariable(i).getName());
                else
                    out.print(data.getVariable(i).getName() + "\t");
            }
            for (int i = 0; i < stab.length; i++) {
                out.print(data.getVariable(i).getName() + "\t");
                for (int j = 0; j < stab[i].length; j++) {
                    if (j == stab[i].length - 1)
                        out.println(stab[i][j]);
                    else
                        out.print(stab[i][j] + "\t");
                }
            }
            out.flush();
            out.close();
            if(tumors)
                 out = new PrintStream("Graph_Relevant_Prior.txt");
            else
                out = new PrintStream("Graph_Relevant_Prior_Normals.txt");
            out.println(g2);
            out.flush();
            out.close();


        }

        if(runBoth)
        {

            System.out.print("Loading Priors...");
            f = new File("prior_sources");
            File [] stuff = f.listFiles();
            ArrayList<File> files = new ArrayList<File>();
            for(int i = 0; i < stuff.length;i++)
            {
                if(!stuff[i].getName().contains("PAM50"))
                    files.add(stuff[i]);
            }
           SparseDoubleMatrix2D [] priors = new SparseDoubleMatrix2D[files.size()];
            for(int i = 0; i < files.size();i++) {
                priors[i] = new SparseDoubleMatrix2D(loadPrior(files.get(i),data.getNumColumns()));
            }

            System.out.println("Done");
            //Basically these are sufficient statistics from the data to evaluate any pathway
            if(useStabilities)
            {
                TetradMatrix stabs = new TetradMatrix(loadStability("Results/Full_Counts_Pathway_Prior.txt",priors[0].rows()));
                if(type!=-1)
                    stabs = new TetradMatrix(loadStability("Results/Full_Counts_Pathway_Prior_" + types[type] + ".txt",priors[0].rows()));
                else if(!tumors)
                    stabs = new TetradMatrix(loadStability("Results/Full_Counts_Pathway_Prior_Normals.txt",priors[0].rows()));
                mgmPriors m  = new mgmPriors(numSub,lambda,stabs,priors);
                m.evaluatePriors();
                double [] weights = m.normalizedExpertWeights;
                double [] pValues = m.pValues;
                double [] normalizedTao = m.normalizedTao;
                PrintStream w = new PrintStream("Weights_Pathway_Prior.txt");
                if(erPositive)
                    w = new PrintStream("Weights_Pathway_Prior_ER_Positive.txt");
                else if(erNegative)
                    w = new PrintStream("Weights_Pathway_Prior_ER_Negative.txt");
                else if(type!=-1)
                    w = new PrintStream("Weights_Pathway_Prior_" + types[type] + ".txt");
                else if(!tumors)
                    w = new PrintStream("Weights_Pathway_Prior_Normals.txt");
                for(int i = 0; i < weights.length;i++)
                {
                    w.println(files.get(i).getName().replace(".txt","") + "\t" + weights[i] + "\t" + pValues[i] + "\t" + normalizedTao[i]);
                }
                w.flush();
                w.close();
            }

            //If you need to recompute all of the edge counts, they will be saved into a matrix
            else {

                System.out.print("Computing Lambda Range...");
                mgmPriors m = new mgmPriors(samps.length, lambda, data, priors, samps);
                System.out.println("Done");
                PrintStream lb = new PrintStream("Pathway_Lambdas.txt");
                if(erPositive)
                    lb = new PrintStream("Pathway_Lambdas_ER_Positive.txt");
                else if(erNegative)
                    lb = new PrintStream("Pathway_Lambdas_ER_Negative.txt");
                else if(type!=-1)
                    lb = new PrintStream("Pathway_Lambdas_" + types[type] + ".txt");
                else if(!tumors)
                    lb = new PrintStream("Pathway_Lambdas_Normals.txt");
                for (int i = 0; i < m.getLambdas().length; i++) {
                    lb.println(Arrays.toString(m.getLambdas()[i]));
                }
                lb.flush();
                lb.close();

                System.out.print("Running Priors...");
                Graph g2 = m.runPriors();
                System.out.println("Done");
                double[][] stab = m.edgeScores;
                double[] weights = m.normalizedExpertWeights;
                double[] pValues = m.pValues;
                double[] normalizedTao = m.normalizedTao;
                double [][] beta = m.betas;
                TetradMatrix fullCounts = m.fullCounts;
                PrintStream w= new PrintStream("Weights_Pathway_Prior.txt");
                if(erPositive)
                    w = new PrintStream("Weights_Pathway_Prior_ER_Positive.txt");
                else if(erNegative)
                    w = new PrintStream("Weights_Pathway_Prior_ER_Negative.txt");
                else if(type!=-1)
                    w = new PrintStream("Weights_Pathway_Prior_" + types[type] + ".txt");
                else if(!tumors)
                    w = new PrintStream("Weights_Pathway_Prior_Normals.txt");
                for (int i = 0; i < weights.length; i++) {
                    w.println(files.get(i).getName().replace(".txt", "") + "\t" + weights[i] + "\t" + pValues[i] + "\t" + normalizedTao[i]);
                }
                w.flush();
                w.close();
                PrintStream out= new PrintStream("Stabilities_Pathway_Prior.txt");
                if(erPositive)
                    out = new PrintStream("Stabilities_Pathway_Prior_ER_Positive.txt");
                else if(erNegative)
                    out = new PrintStream("Stabilities_Pathway_Prior_ER_Negative.txt");
                else if(type!=-1)
                    out = new PrintStream("Stabilities_Pathway_Prior_" + types[type] + ".txt");
                else if(!tumors)
                    out = new PrintStream("Stabilities_Pathway_Prior_Normals.txt");
                for (int i = 0; i < data.getNumColumns(); i++) {
                    if (i == data.getNumColumns() - 1)
                        out.println(data.getVariable(i).getName());
                    else
                        out.print(data.getVariable(i).getName() + "\t");
                }
                for (int i = 0; i < stab.length; i++) {
                    out.print(data.getVariable(i).getName() + "\t");
                    for (int j = 0; j < stab[i].length; j++) {
                        if (j == stab[i].length - 1)
                            out.println(stab[i][j]);
                        else
                            out.print(stab[i][j] + "\t");
                    }
                }
                out.flush();
                out.close();
                out = new PrintStream("Graph_Pathway_Prior.txt");
                if(erPositive)
                    out = new PrintStream("Graph_Pathway_Prior_ER_Positive.txt");
                else if(erNegative)
                    out = new PrintStream("Graph_Pathway_Prior_ER_Negative.txt");
                else if(type!=-1)
                    out = new PrintStream("Graph_Pathway_Prior_" + types[type] + ".txt");
                else if(!tumors)
                    out = new PrintStream("Graph_Pathway_Prior_Normals.txt");
                out.println(g2);
                out.flush();
                out.close();
                out = new PrintStream("Full_Counts_Pathway_Prior.txt");
                if(erPositive)
                    out = new PrintStream("Full_Counts_Pathway_Prior_ER_Positive.txt");
                else if(erNegative)
                    out = new PrintStream("Full_Counts_Pathway_Prior_ER_Negative.txt");
                else if(type!=-1)
                    out = new PrintStream("Full_Counts_Pathway_Prior_" + types[type] + ".txt");
                else if(!tumors)
                    out = new PrintStream("Full_Counts_Pathway_Prior_Normals.txt");
                for(int i =0; i < fullCounts.rows();i++)
                {
                    for(int j = 0; j < fullCounts.columns();j++)
                    {
                        if(j==fullCounts.columns()-1)
                            out.println(fullCounts.get(i,j));
                        else
                            out.print(fullCounts.get(i,j) + "\t");
                    }
                }

                out = new PrintStream("Beta_Pathway_Prior.txt");
                if(erPositive)
                    out = new PrintStream("Beta_Pathway_Prior_ER_Positive.txt");
                else if(erNegative)
                    out = new PrintStream("Beta_Pathway_Prior_ER_Negative.txt");
                else if (type!=-1)
                    out = new PrintStream("Beta_Pathway_Prior_" + types[type] + ".txt");
                else if(!tumors)
                    out = new PrintStream("Beta_Pathway_Prior_Normals.txt");

                DataSet cont = MixedUtils.getContinousData(data);
                for (int i = 0; i < cont.getNumColumns(); i++) {
                    if (i == cont.getNumColumns() - 1)
                        out.println(cont.getVariable(i).getName());
                    else
                        out.print(cont.getVariable(i).getName() + "\t");
                }
                for (int i = 0; i < beta.length; i++) {
                    out.print(cont.getVariable(i).getName() + "\t");
                    for (int j = 0; j < beta[0].length; j++) {
                        if (j == beta[0].length - 1)
                            out.println(beta[i][j]);
                        else
                            out.print(beta[i][j] + "\t");
                    }
                }

                out.flush();
                out.close();
            }

        }


        //Output metrics, precision and recall of PAM50 Genes (Check the neighborhood of the Subtype variable)
        //, Recovery of relevant pathway connections
    }

    public static double [][] loadPAM50(File data, int numVariables) throws Exception
    {
        double [][] temp = new double[numVariables][numVariables];
        BufferedReader b = new BufferedReader(new FileReader(data));
        for(int i = 0; i < numVariables;i++)
        {
            String [] line = b.readLine().split("\t");
            for(int j = 0; j < numVariables;j++)
            {
                temp[i][j] = Double.parseDouble(line[j]);
            }
        }
        return temp;
    }
    public static double [][] loadPrior(File data, int numVariables) throws Exception
    {
        double [][] temp = new double[numVariables][numVariables];
        BufferedReader b = new BufferedReader(new FileReader(data));
        b.readLine();
        for(int i = 0; i < numVariables;i++)
        {
            String [] line = b.readLine().split("\t");
            for(int j = 0; j < numVariables;j++)
            {
                temp[i][j] = Double.parseDouble(line[j]);
            }
        }
        return temp;
    }
    public static double [][] loadStability(String data, int numVariables) throws Exception
    {

        double [][] temp = new double[numVariables][numVariables];
        BufferedReader b = new BufferedReader(new FileReader(new File(data)));
        for(int i = 0; i < numVariables;i++)
        {
            String [] line = b.readLine().split("\t");
            for(int j = 0; j < numVariables;j++)
            {
                temp[i][j] = Double.parseDouble(line[j]);
            }
        }
        return temp;
    }
}
