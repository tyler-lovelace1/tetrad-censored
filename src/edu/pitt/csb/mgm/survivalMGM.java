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

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.jet.math.Functions;
import edu.cmu.tetrad.data.CensoredVariable;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.EdgeListGraph;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.search.GraphSearch;
import edu.cmu.tetrad.search.IndependenceTest;
import edu.cmu.tetrad.util.Function;
import edu.cmu.tetrad.util.StatUtils;

import java.util.ArrayList;
import java.util.List;

//import cern.colt.Arrays;
//import la.matrix.Matrix;
//import la.matrix.DenseMatrix;
//import ml.optimization.AcceleratedProximalGradient;
//import ml.optimization.ProximalMapping;
//import ml.utils.Matlab;

/**
 * Implementation of Lee and Hastie's (2012) pseudolikelihood method for learning
 * Mixed Gaussian-Categorical Graphical Models
 * Created by ajsedgewick on 7/15/15.
 */
public class survivalMGM extends ConvexProximal implements GraphSearch{
    private DoubleFactory2D factory2D = DoubleFactory2D.dense;
    private DoubleFactory1D factory1D = DoubleFactory1D.dense;

    //private DoubleFactory2D factory2D = DoubleFactory2D.sparse;
    //private DoubleFactory1D factory1D = DoubleFactory1D.sparse;

    //Continuous Data
    private DoubleMatrix2D xDat;

    //Discrete Data coded as integers, no IntMatrix2D apparently...
    private DoubleMatrix2D yDat;
    
    //Survival Time Data
    private DoubleMatrix2D tDat;
    
    //Survival Censorship Data
    private DoubleMatrix2D cDat;
    
    //Ranking of Survival Time
    private int[][] order;

    //Frequency for each unique survival time
    private int[][] H;

    //Survival Time Data
    private double[][][] mu_t;

    //Survival Time Data
    private double[][][] var_t;

    //Survival Time Data
    private double[][][] p_t;

    private List<Node> variables;
    private List<Node> initVariables = null;

    //Discrete Data coded as dummy variables
    private DoubleMatrix2D dDat;

    private DoubleMatrix1D lambda;
    private Algebra alg = new Algebra();

    private long elapsedTime = 0;

    //Levels of Discrete variables
    private int[] l;
    private int lsum;
    private int[] lcumsum;
    //Number of events
    private int[] Ne;
    int p;
    int q;
    int r;
    int n;
    private long timeout = -1;

    //parameter weights
    private DoubleMatrix1D weights;
    private DoubleMatrix2D censoredWeights;


    public double timePerIter = 0;
    public int iterCount = 0;

    public survivalMGM(DoubleMatrix2D x, DoubleMatrix2D y, DoubleMatrix2D t, DoubleMatrix2D c, List<Node> variables, int[] l, double[] lambda){

        if(l.length != y.columns())
            throw new IllegalArgumentException("length of l doesn't match number of variables in Y");

        if((y.rows() != x.rows()) || (x.rows() != t.rows()) || (x.rows() != c.rows()) || (y.rows() != t.rows()) || (y.rows() != c.rows()) || (t.rows() != c.rows()))
            throw new IllegalArgumentException("different number of samples for x, y, or c");
        
        if(t.columns() != c.columns())
            throw new IllegalArgumentException("different number of survival time and censorship variables");

        //lambda should have 5 values corresponding to cc, cd, dd, sc, and sd
        if(lambda.length != 5)
            throw new IllegalArgumentException("Lambda should have three values for cc, cd, dd, sc, and sd edges respectively");


        this.xDat = x;
        this.yDat = y;
        this.l = l;
        this.p = x.columns();
        this.q = y.columns();
        this.r = t.columns();
        this.n = x.rows();
        this.variables = variables;
        
        this.order = new int[n][r];
        this.H = new int[n][r];
        this.Ne = new int[r];

        for (int i = 0; i < c.columns(); i++) this.Ne[i] = (int) c.viewColumn(i).zSum();

        this.lambda = factory1D.make(lambda);
        fixData();
        initParameters();
        makeDummy();
        calcWeights();
    }

    public survivalMGM(DataSet ds, double[] lambda){
    	this.variables = ds.getVariables();

        DataSet dsCont = MixedUtils.getContinousData(ds);
        DataSet dsDisc = MixedUtils.getDiscreteData(ds);
        DataSet dsSurv = MixedUtils.getCensoredData(ds);

//        int tempq = dsDisc.getVariables().size();
//        for (Node n : dsDisc.getVariables()) {
////        	if (n.getName().toLowerCase().contains("censor")) {
////        		dsDisc.removeColumn(n);
////        	}
//            if (n == dsDisc.getVariable(tempq-1)) {
//                dsCont.addVariable(n);
//                dsDisc.removeColumn(n);
//            }
//            //System.out.println("discVar: " + n.getName());
//        }
//        for (Node n : dsCont.getVariables()) {
//            System.out.println("contVar: " + n.getName());
//        }
//        for (Node n : dsDisc.getVariables()) {
//            System.out.println("discVar: " + n.getName());
//        }
        this.xDat = factory2D.make(dsCont.getDoubleData().toArray());
        this.yDat = factory2D.make(dsDisc.getDoubleData().toArray());
        this.tDat = factory2D.make(dsSurv.getDoubleData().toArray());
//        this.cDat = factory2D.make(dsSurv.getDoubleCensorData().toArray());
        this.cDat = factory2D.make(dsSurv.getNumRows(), dsSurv.getNumColumns(), 0.0);
        this.l = MixedUtils.getDiscLevels(ds);
        this.p = xDat.columns();
        this.q = yDat.columns();
        this.r = tDat.columns();
        this.n = xDat.rows();

        this.order = new int[n][r];
        this.H = new int[n][r];
        this.Ne = new int[r];

        for (int j = 0; j < r; j++) {
            for (int i = 0; i < n; i++) {
                cDat.set(i, j, ((CensoredVariable) dsSurv.getVariable(j)).getCensor(i));
                order[i][j] = ((CensoredVariable) dsSurv.getVariable(j)).getOrder(i);
                try {
                    H[i][j] = ((CensoredVariable) dsSurv.getVariable(j)).getH(i);
                } catch (Exception e) {
                    H[i][j] = -1;
                }
            }

            this.Ne[j] = (int) cDat.viewColumn(j).zSum();

            int count = 0;
            int Hsum = 0;
            while (H[count][j] >= 0) {
//                System.out.println(H[count][i]);
                Hsum += H[count][j];
                count++;
                if (count >= this.n)
                    break;
            }
//            System.out.println(Hsum);
            if (Hsum != this.n) {
                throw new IllegalArgumentException("Hsum " + Hsum + " does not equal n " + this.n);
            }
        }

//        System.out.println(this.p);
//        System.out.println(this.q);

        //the variables are now ordered continuous first then discrete
        this.variables = new ArrayList<Node>();
        variables.addAll(dsCont.getVariables());
        variables.addAll(dsDisc.getVariables());
        variables.addAll(dsSurv.getVariables());
        // variables.addAll(dsCen.getVariables());

        this.initVariables = ds.getVariables();
        
        // System.out.println(this.initVariables);

        this.lambda = factory1D.make(lambda);

        //Data is checked for 0 or 1 indexing and for missing levels and N(0,1) Standardizes continuous data
        fixData();

        //Initialize all parameters to zeros
        initParameters();

        //Creates dummy variables for each category of discrete variables (stored in dDat)
        makeDummy();

        //Sets continuous variable weights to standard deviation and discrete variable weights to p*(1-p) for each category
        calcWeights();
    }

    public static class survivalMGMParams{
        //Model parameters
        private DoubleMatrix2D beta; //continuous-continuous
        private DoubleMatrix1D betad; //cont squared node pot
        private DoubleMatrix2D theta; //continuous-discrete
        private DoubleMatrix2D phi; //discrete-discrete
        private DoubleMatrix1D alpha1; //cont linear node pot
        private DoubleMatrix1D alpha2; //disc node pot
        private DoubleMatrix2D gamma; //continuous-survival
        private DoubleMatrix2D eta; //discrete-survival

        public survivalMGMParams(){

        }

        //nothing is copied here, all pointers back to inputs...
        public survivalMGMParams(DoubleMatrix2D beta, DoubleMatrix1D betad, DoubleMatrix2D theta,
            DoubleMatrix2D phi, DoubleMatrix1D alpha1, DoubleMatrix1D alpha2, DoubleMatrix2D gamma,
            DoubleMatrix2D eta) {
            this.beta = beta;
            this.betad = betad;
            this.theta = theta;
            this.phi = phi;
            this.alpha1 = alpha1;
            this.alpha2 = alpha2;
            this.gamma = gamma;
            this.eta = eta;
        }

        //copy from another parameter set
        public survivalMGMParams(survivalMGMParams parIn){
            this.beta = parIn.beta.copy();
            this.betad = parIn.betad.copy();
            this.theta = parIn.theta.copy();
            this.phi = parIn.phi.copy();
            this.alpha1 = parIn.alpha1.copy();
            this.alpha2 = parIn.alpha2.copy();
            this.gamma = parIn.gamma.copy();
            this.eta = parIn.eta.copy();
        }

        //copy params from flattened vector
        public survivalMGMParams(DoubleMatrix1D vec, int p, int ltot, int r){
            int[] lens = {p*p, p, p*ltot, ltot*ltot, p, ltot, p*r, r*ltot};
            int[] lenSums = new int[lens.length];
            lenSums[0] = lens[0];
            for(int i = 1; i < lenSums.length; i++){
                lenSums[i] = lens[i] + lenSums[i-1];
            }

            if(vec.size() != lenSums[7])
                throw new IllegalArgumentException("Param vector dimension doesn't match: Found " + vec.size() + " need " + lenSums[5]);

            beta = DoubleFactory2D.dense.make(vec.viewPart(0, lens[0]).toArray(), p);
            betad = vec.viewPart(lenSums[0], lens[1]).copy();
            theta = DoubleFactory2D.dense.make(vec.viewPart(lenSums[1], lens[2]).toArray(), ltot);
            phi = DoubleFactory2D.dense.make(vec.viewPart(lenSums[2], lens[3]).toArray(), ltot);
            alpha1 = vec.viewPart(lenSums[3], lens[4]).copy();
            alpha2 = vec.viewPart(lenSums[4], lens[5]).copy();
            gamma = DoubleFactory2D.dense.make(vec.viewPart(lenSums[5], lens[6]).toArray(), p);
            eta = DoubleFactory2D.dense.make(vec.viewPart(lenSums[6], lens[7]).toArray(), ltot);
        }

        public String toString(){
            String outStr = "alpha1: " + alpha1.toString();
            outStr += "\nalpha2: " + alpha2.toString();
            outStr += "\nbeta: " + beta.toString();
            outStr += "\nbetad: " + betad.toString();
            outStr += "\ntheta: " + theta.toString();
            outStr += "\nphi: " + phi.toString();
            outStr += "\ngamma: " + gamma.toString();
            outStr += "\neta: " + eta.toString();
            return outStr;
        }

        public DoubleMatrix1D getAlpha1() {
            return alpha1;
        }

        public DoubleMatrix1D getAlpha2() {
            return alpha2;
        }

        public DoubleMatrix1D getBetad() {
            return betad;
        }

        public DoubleMatrix2D getBeta() {
            return beta;
        }

        public DoubleMatrix2D getPhi() {
            return phi;
        }

        public DoubleMatrix2D getTheta() {
            return theta;
        }
        
        public DoubleMatrix2D getGamma() {
            return gamma;
        }
        
        public DoubleMatrix2D getEta() {
            return eta;
        }

        public void setAlpha1(DoubleMatrix1D alpha1) {
            this.alpha1 = alpha1;
        }

        public void setAlpha2(DoubleMatrix1D alpha2) {
            this.alpha2 = alpha2;
        }

        public void setBeta(DoubleMatrix2D beta) {
            this.beta = beta;
        }

        public void setBetad(DoubleMatrix1D betad) {
            this.betad = betad;
        }

        public void setPhi(DoubleMatrix2D phi) {
            this.phi = phi;
        }

        public void setTheta(DoubleMatrix2D theta) {
            this.theta = theta;
        }
        
        public void setGamma(DoubleMatrix2D gamma) {
            this.gamma = gamma;
        }
        
        public void setEta(DoubleMatrix2D eta) {
            this.eta = eta;
        }

        /**
         * Copy all params into a single vector
         * @return
         */
        public DoubleMatrix1D toMatrix1D(){
        	// System.out.println(gamma instanceof DoubleMatrix2D);
        	// System.out.println(gamma);
            DoubleFactory1D fac = DoubleFactory1D.dense;
            int p = alpha1.size();
            int r;
            if (gamma instanceof DoubleMatrix2D) {
            	r = gamma.columns();
            } else {
            	r = 1;
            }
            int ltot = alpha2.size();
            int[] lens = {p*p, p, p*ltot, ltot*ltot, p, ltot, r*p, r*ltot};
            int[] lenSums = new int[lens.length];
            lenSums[0] = lens[0];
            for(int i = 1; i < lenSums.length; i++){
                lenSums[i] = lens[i] + lenSums[i-1];
            }

            DoubleMatrix1D outVec = fac.make(p*p + p + p*ltot + ltot*ltot + p + ltot + r*p + r*ltot);
            outVec.viewPart(0, lens[0]).assign(flatten(beta));
            outVec.viewPart(lenSums[0],lens[1]).assign(betad);
            outVec.viewPart(lenSums[1],lens[2]).assign(flatten(theta));
            outVec.viewPart(lenSums[2],lens[3]).assign(flatten(phi));
            outVec.viewPart(lenSums[3],lens[4]).assign(alpha1);
            outVec.viewPart(lenSums[4],lens[5]).assign(alpha2);
            if (r > 1) {
            	outVec.viewPart(lenSums[5],lens[6]).assign(flatten(gamma));
            	outVec.viewPart(lenSums[6],lens[7]).assign(flatten(eta));
            } else {
            	outVec.viewPart(lenSums[5],lens[6]).assign(gamma.viewColumn(0));
            	outVec.viewPart(lenSums[6],lens[7]).assign(eta.viewColumn(0));
            }

            return outVec;
        }

        //likely depreciated
        public double[][] toVector(){
            double[][] outArr = new double[1][];
            outArr[0] = toMatrix1D().toArray();
            return outArr;
        }
    }

    private survivalMGMParams params;

    public void setParams(survivalMGMParams newParams){
        params = newParams;
    }

    //create column major vector from matrix (i.e. concatenate columns)
    public static DoubleMatrix1D flatten(DoubleMatrix2D m){
        DoubleMatrix1D[] colArray = new DoubleMatrix1D[m.columns()];
        for(int i = 0; i < m.columns(); i++){
            colArray[i] = m.viewColumn(i);
        }

        return DoubleFactory1D.dense.make(colArray);
    }

    //init all parameters to zeros except for betad which is set to 1s
    private void initParameters(){
        lcumsum = new int[l.length+1];
        lcumsum[0] = 0;
        for(int i = 0; i < l.length; i++){
            lcumsum[i+1] = lcumsum[i] + l[i];
        }
        lsum = lcumsum[l.length];

        //LH init to zeros, maybe should be random init?
        DoubleMatrix2D beta = factory2D.make(xDat.columns(), xDat.columns(), 0.0); //continuous-continuous
        DoubleMatrix1D betad = factory1D.make(xDat.columns(), 1.0); //cont squared node pot
        DoubleMatrix2D  theta = factory2D.make(lsum, xDat.columns(), 0.0);; //continuous-discrete
        DoubleMatrix2D phi = factory2D.make(lsum, lsum, 0.0); //discrete-discrete
        DoubleMatrix1D alpha1 = factory1D.make(xDat.columns(), 0.0); //cont linear node pot
        DoubleMatrix1D alpha2 = factory1D.make(lsum, 0.0); //disc node potbeta =
        DoubleMatrix2D gamma = factory2D.make(xDat.columns(), tDat.columns(), 0.0);
        DoubleMatrix2D eta = factory2D.make(lsum, tDat.columns(), 0.0);
        params = new survivalMGMParams(beta, betad, theta, phi, alpha1, alpha2, gamma, eta);

        //separate lambda for each type of edge, [cc, cd, dd]
        //lambda = factory1D.make(3);
    }

    // avoid underflow in log(sum(exp(x))) calculation
    private double logsumexp(DoubleMatrix1D x){
        DoubleMatrix1D myX = x.copy();
        double maxX = StatUtils.max(myX.toArray());
        return Math.log(myX.assign(Functions.minus(maxX)).assign(Functions.exp).zSum()) + maxX;
    }

    //calculate parameter weights as in Lee and Hastie
    private void calcWeights(){
        weights = factory1D.make(p+q+r, 0);
        censoredWeights = factory2D.make(p+q, r, 1);
        //weightsByCat = factory1D.make(p+lsum);
        for(int i = 0; i < p; i++){
            weights.set(i, StatUtils.sd(xDat.viewColumn(i).toArray()));
        }
        //Continuous variable weights are standard deviations


        //Discrete variable weights for each variable-category pair are p(1-p) where p is the percentage of times that category appears
//        System.out.println(yDat);
//        int count = 0;
        for(int j = 0; j < q; j++){
            double curWeight = 0;
            for(int k = 0; k < l[j] ; k++){
                double curp = yDat.viewColumn(j).copy().assign(Functions.equals(k+1)).zSum()/(double) n;
                curWeight += curp*(1-curp);
            }
            weights.set(p+j, Math.sqrt(curWeight));
        }

        int maxNe = 0;
        for (int k = 0; k < r; k++) {
            if (Ne[k] > maxNe) maxNe = Ne[k];
        }

//        System.out.println("maxNe: " + maxNe);

        mu_t = new double[maxNe][r][p];
        var_t = new double[maxNe][r][p];
        p_t = new double[maxNe][r][lsum];

        int e_count;
        DoubleMatrix1D tempVec;
        int idx;
        for (int k = 0; k < r; k++) {
//            System.out.println("censored variable " + k);
            e_count = 0;
            for (int i = 0; i < n; i++) {
                if (cDat.get(order[i][k], k) == 1) {
                    idx = i;
                    while ((idx-1 > 0) && (tDat.get(order[idx][k], k) == tDat.get(order[idx-1][k], k))){
                        idx--;
                    }
                    for (int j = 0; j < p; j++) {
                        for (int m = idx; m < n; m++) {
                            mu_t[e_count][k][j] += xDat.get(m, j);
                            var_t[e_count][k][j] += xDat.get(m, j) * xDat.get(m, j);
                        }
                        mu_t[e_count][k][j] /= (n-idx);
                        var_t[e_count][k][j] /= (n-idx);
                        var_t[e_count][k][j] -= mu_t[e_count][k][j] * mu_t[e_count][k][j];
                        var_t[e_count][k][j] *= ((n-idx) / (double) (n-idx-1));
                    }

                    tempVec = factory1D.make(lsum, 0);

                    for (int m = idx; m < n; m++) {
                        tempVec.assign(dDat.viewRow(m), Functions.plus);
                    }

                    tempVec.assign(Functions.div(n-idx));

                    for (int m = 0; m < lsum; m++) {
                        p_t[e_count][k][m] = tempVec.get(m);
                    }

                }

                e_count++;
                if (e_count == Ne[k]) break;
//                    System.out.println("event " + e_count);
            }
        }

    }

    private void calcCensoredWeights(survivalMGMParams par) {
        DoubleMatrix1D tempVec;
        DoubleMatrix1D expEta;
        int e_count;

        censoredWeights.assign(0);

        for (int k = 0; k < r; k++) {
            e_count = 0;
            expEta = par.eta.viewColumn(k).copy();
            expEta.assign(Functions.exp);

            for (int i = 0; i < n; i++) {
                if (cDat.get(order[i][k], k) == 1) {
                    tempVec = xDat.viewRow(order[i][k]).copy().assign(factory1D.make(mu_t[e_count][k]).assign(factory1D.make(var_t[e_count][k]).assign(par.gamma.viewColumn(k).copy(), Functions.mult), Functions.plus), Functions.minus);

                    censoredWeights.viewColumn(k).viewPart(0, p).assign(tempVec.assign(Functions.square), Functions.plus);

                    tempVec = dDat.viewRow(order[i][k]).copy().assign(factory1D.make(p_t[e_count][k]).assign(expEta, Functions.mult).assign(factory1D.make(p_t[e_count][k]).assign(expEta, Functions.mult).assign(factory1D.make(p_t[e_count][k]).assign(Functions.plus(1)), Functions.minus), Functions.div), Functions.minus);
                    tempVec.assign(Functions.square);

                    for (int j = 0; j < q; j++) {
                        censoredWeights.viewColumn(k).viewPart(p+j, 1).assign(Functions.plus(tempVec.viewPart(lcumsum[j], l[j]).zSum()));
                    }

                    e_count++;
                }

                if (e_count == Ne[k]) break;
            }
            censoredWeights.viewColumn(k).assign(Functions.div(Ne[k]));
        }

        censoredWeights.assign(Functions.sqrt);

//        System.out.println("gamma weights:\n" + censoredWeights.viewPart(0,0,p,r).viewDice() + "\n");
//        System.out.println("eta weights:\n" + censoredWeights.viewPart(p,0,q,r).viewDice() + "\n");
    }

    /**
     * Convert discrete data (in yDat) to a matrix of dummy variables (stored in dDat)
     */
    private void makeDummy(){
        dDat = factory2D.make(n, lsum);
        for(int i = 0; i < q; i++){
            for(int j = 0; j < l[i]; j++){
                DoubleMatrix1D curCol = yDat.viewColumn(i).copy().assign(Functions.equals(j+1));
                if(curCol.zSum() == 0)
                    throw new IllegalArgumentException("Discrete data is missing a level: variable " + yDat.viewColumn(i) + " level " + j);
                dDat.viewColumn(lcumsum[i]+j).assign(curCol);
            }
        }
//        DoubleMatrix1D catVar = factory1D.make(lsum);
//        double p;
//        double minVar = 1.0;
//        for (int i = 0; i < lsum; i++) {
//            p = dDat.viewColumn(i).zSum() / ((double) n);
//            catVar.set(i, p*(1-p));
//            if (minVar > p*(1-p))
//                minVar = p*(1-p);
//        }
//        // System.out.println(Thread.currentThread().getName() + " Category Variances: " + catVar);
//        System.out.println("    " + Thread.currentThread().getName() + " Average Variance: " + catVar.zSum() / ((double) lsum));
//        System.out.println("    " + Thread.currentThread().getName() + " Minimum Variance: " + minVar);
    }

    /**
     * checks if yDat is zero indexed and converts to 1 index. zscores x
     */
    private void fixData(){
        double ymin = StatUtils.min(flatten(yDat).toArray());
        if(ymin < 0 || ymin > 1)
            throw new IllegalArgumentException("Discrete data must be either zero or one indexed. Found min index: " + ymin);

        if(ymin==0){
            yDat.assign(Functions.plus(1.0));
        }

        //z-score columns of X
        for(int i = 0; i < p; i++){
            xDat.viewColumn(i).assign(StatUtils.standardizeData(xDat.viewColumn(i).toArray()));
        }

    }

    /**
     * non-penalized -log(pseudolikelihood) this is the smooth function g(x) in prox gradient
     *
     * @param parIn
     * @return
     */
    public double smoothValue(DoubleMatrix1D parIn){
        //work with copy
        survivalMGMParams par = new survivalMGMParams(parIn, p, lsum, r);

        for(int i = 0; i < par.betad.size(); i++){
            if(par.betad.get(i)<=0)
                return Double.POSITIVE_INFINITY;
        }
        //double nll = 0;
        //int n = xDat.rows();
        //beta=beta+beta';
        //phi=phi+phi';
        upperTri(par.beta, 1);
        par.beta.assign(alg.transpose(par.beta), Functions.plus);

        for(int i = 0; i < q; i++){
            par.phi.viewPart(lcumsum[i], lcumsum[i], l[i], l[i]).assign(0);
        }
        // ensure mats are upper triangular
        upperTri(par.phi,0);
        par.phi.assign(alg.transpose(par.phi), Functions.plus);


        //Xbeta=X*beta*diag(1./betad);
        DoubleMatrix2D divBetaD = factory2D.diagonal(factory1D.make(p,1.0).assign(par.betad, Functions.div));
        DoubleMatrix2D xBeta = alg.mult(xDat,alg.mult(par.beta, divBetaD));

        //Dtheta=D*theta*diag(1./betad);
        DoubleMatrix2D dTheta = alg.mult(alg.mult(dDat, par.theta), divBetaD);

        // Squared loss
        //sqloss=-n/2*sum(log(betad))+...
        //.5*norm((X-e*alpha1'-Xbeta-Dtheta)*diag(sqrt(betad)),'fro')^2;
        DoubleMatrix2D tempLoss = factory2D.make(n, xDat.columns());

        //wxprod=X*(theta')+D*phi+e*alpha2';
        DoubleMatrix2D wxProd = alg.mult(xDat, alg.transpose(par.theta));
        wxProd.assign(alg.mult(dDat, par.phi), Functions.plus);
        for(int i = 0; i < n; i++){
            for(int j = 0; j < xDat.columns(); j++){
                tempLoss.set(i,j,xDat.get(i,j) - par.alpha1.get(j) - xBeta.get(i,j) - dTheta.get(i,j));
            }
            for(int j = 0; j < dDat.columns(); j++){
                wxProd.set(i,j,wxProd.get(i,j) + par.alpha2.get(j));
            }
        }

        double sqloss = -n/2.0*par.betad.copy().assign(Functions.log).zSum() +
                .5 * Math.pow(alg.normF(alg.mult(tempLoss, factory2D.diagonal(par.betad.copy().assign(Functions.sqrt)))), 2);


        // categorical loss
        /*catloss=0;
        wxprod=X*(theta')+D*phi+e*alpha2'; %this is n by Ltot
        for r=1:q
            wxtemp=wxprod(:,Lsum(r)+1:Lsum(r)+L(r));
            denom= logsumexp(wxtemp,2); %this is n by 1
            catloss=catloss-sum(wxtemp(sub2ind([n L(r)],(1:n)',Y(:,r))));
            catloss=catloss+sum(denom);
        end
        */

        double catloss = 0;
        for(int i = 0; i < yDat.columns(); i++){
            DoubleMatrix2D wxTemp = wxProd.viewPart(0, lcumsum[i], n, l[i]);
            for(int k = 0; k < n; k++){
                DoubleMatrix1D curRow = wxTemp.viewRow(k);

                catloss -= curRow.get((int) yDat.get(k, i) - 1);
                catloss += logsumexp(curRow);
            }
        }
        /*
        for (int i = 0; i < q; i++)
        	par.eta.viewRow(lcumsum[i]).assign(0.0);
        */
        /*
         * Cox regression loss
         */

        double coxloss = 0.0;
        double tempcoxloss;
        double rs_sum;
        double Hsum;
        double Hsumtemp;
        double sub;
        DoubleMatrix1D temp = factory1D.make(n, 0.0);

//        System.out.println("gamma: " + par.gamma);
//        System.out.println("eta: " + par.eta);
        for (int i = 0; i < r; i++) {
            temp.assign(alg.mult(xDat, par.gamma.viewColumn(i)));
            temp.assign(alg.mult(dDat, par.eta.viewColumn(i)), Functions.plus);
            temp.assign(Functions.exp);

            DoubleMatrix1D sTemp = temp.copy();

            for (int j = 0; j < n; j++)
                sTemp.set(j, temp.get(order[j][i]));

            temp.assign(sTemp);

            tempcoxloss = 0.0;
            rs_sum = temp.zSum();
            int j = 0;
            int idx = 0;
            int m;
            while (idx < n) {
//                System.out.println(idx + " : " + H[j][i]);
                Hsum = 0.0;
                Hsumtemp = 0.0;
                m = 0;
                sub = 0;
                for (int k = 0; k < H[j][i]; k++) {
                    m += cDat.get(order[idx+k][i], i);
                    Hsum += cDat.get(order[idx+k][i], i) * (alg.mult(xDat.viewRow(order[idx+k][i]), par.gamma.viewColumn(i)) + alg.mult(dDat.viewRow(order[idx+k][i]), par.eta.viewColumn(i)));
                    Hsumtemp += cDat.get(order[idx+k][i], i) * temp.get(idx+k);
                    sub += temp.get(idx+k);
                }

                if ((sub > rs_sum) || (rs_sum <= 0)){
                    rs_sum = temp.viewPart(idx, n-idx).zSum();
                }
                assert Hsumtemp <= sub : "Hsumtemp > sub";
                assert sub <= rs_sum : "sub > rs_sum";
                assert rs_sum >= 0.0 : "rs_sum < 0";
                assert idx < n : "idx >= n";

//                System.out.println("m: " + m);
//                System.out.println("Hsum: " + Hsum);
//                System.out.println("Hsumtemp: " + Hsumtemp);
//                System.out.println("sub: " + sub);
//                System.out.println("rs_sum: " + rs_sum);

                tempcoxloss -= Hsum;
                for (int l = 0; l < m; l++) {
                    tempcoxloss += Math.log(rs_sum - l / ((double) m) * Hsumtemp);
                }

                idx += H[j][i];
                j++;
                rs_sum -= sub;
            }
//            System.out.println("tempcoxloss: " + tempcoxloss);
//            System.out.println("coxloss: " + coxloss);
//            System.out.println("Ne: " + (double) Ne[i]);
            coxloss += tempcoxloss / (double) Ne[i];
        }
//        System.out.println("coxloss: " + coxloss);
        return (sqloss + catloss)/((double) n)  + coxloss;
    }

    /**
     * non-penalized -log(pseudolikelihood) this is the smooth function g(x) in prox gradient
     * this overloaded version calculates both nll and the smooth gradient at the same time
     * any value in gradOut will be replaced by the new calculations
     *
     *
     * @param parIn
     * @param gradOutVec
     * @return
     */
    public double smooth(DoubleMatrix1D parIn, DoubleMatrix1D gradOutVec){
        //work with copy
    	survivalMGMParams par = new survivalMGMParams(parIn, p, lsum, r);

    	survivalMGMParams gradOut = new survivalMGMParams();

        for(int i = 0; i < par.betad.size(); i++){
            if(par.betad.get(i)<0)
                return Double.POSITIVE_INFINITY;
        }

        //beta=beta-diag(diag(beta));
        //for r=1:q
        //  phi(Lsum(r)+1:Lsum(r+1),Lsum(r)+1:Lsum(r+1))=0;
        //end
        //beta=triu(beta); phi=triu(phi);
        //beta=beta+beta';
        //phi=phi+phi';
        upperTri(par.beta, 1);
        par.beta.assign(alg.transpose(par.beta), Functions.plus);

        for(int i = 0; i < q; i++){
            par.phi.viewPart(lcumsum[i], lcumsum[i], l[i], l[i]).assign(0);
        }
        //ensure matrix is upper triangular
        upperTri(par.phi,0);
        par.phi.assign(alg.transpose(par.phi), Functions.plus);

        //Xbeta=X*beta*diag(1./betad);
        DoubleMatrix2D divBetaD = factory2D.diagonal(factory1D.make(p,1.0).assign(par.betad, Functions.div));
        DoubleMatrix2D xBeta = alg.mult(xDat,alg.mult(par.beta, divBetaD));

        //Dtheta=D*theta*diag(1./betad);
        DoubleMatrix2D dTheta = alg.mult(alg.mult(dDat, par.theta), divBetaD);

        // Squared loss
        //tempLoss =  (X-e*alpha1'-Xbeta-Dtheta) = -res (in gradient code)
        DoubleMatrix2D tempLoss = factory2D.make(n, xDat.columns());

        //wxprod=X*(theta')+D*phi+e*alpha2';
        DoubleMatrix2D wxProd = alg.mult(xDat, alg.transpose(par.theta));
        wxProd.assign(alg.mult(dDat, par.phi), Functions.plus);
        for(int i = 0; i < n; i++){
            for(int j = 0; j < xDat.columns(); j++){
                tempLoss.set(i,j,xDat.get(i,j) - par.alpha1.get(j) - xBeta.get(i,j) - dTheta.get(i,j));
            }
            for(int j = 0; j < dDat.columns(); j++){
                wxProd.set(i,j,wxProd.get(i,j) + par.alpha2.get(j));
            }
        }

        //sqloss=-n/2*sum(log(betad))+...
        //.5*norm((X-e*alpha1'-Xbeta-Dtheta)*diag(sqrt(betad)),'fro')^2;
        double sqloss = -n/2.0*par.betad.copy().assign(Functions.log).zSum() +
                .5 * Math.pow(alg.normF(alg.mult(tempLoss, factory2D.diagonal(par.betad.copy().assign(Functions.sqrt)))), 2);

        //ok now tempLoss = res
        tempLoss.assign(Functions.mult(-1));

        //gradbeta=X'*(res);
        gradOut.beta = alg.mult(alg.transpose(xDat), tempLoss);

        //gradbeta=gradbeta-diag(diag(gradbeta)); % zero out diag
        //gradbeta=tril(gradbeta)'+triu(gradbeta);
        DoubleMatrix2D lowerBeta = alg.transpose(lowerTri(gradOut.beta.copy(), -1));
        upperTri(gradOut.beta, 1).assign(lowerBeta, Functions.plus);

        //gradalpha1=diag(betad)*sum(res,1)';
        gradOut.alpha1 = alg.mult(factory2D.diagonal(par.betad),margSum(tempLoss, 1));

        //gradtheta=D'*(res);
        gradOut.theta = alg.mult(alg.transpose(dDat), tempLoss);

        // categorical loss
        /*catloss=0;
        wxprod=X*(theta')+D*phi+e*alpha2'; %this is n by Ltot
        for r=1:q
            wxtemp=wxprod(:,Lsum(r)+1:Lsum(r)+L(r));
            denom= logsumexp(wxtemp,2); %this is n by 1
            catloss=catloss-sum(wxtemp(sub2ind([n L(r)],(1:n)',Y(:,r))));
            catloss=catloss+sum(denom);
        end
        */

        double catloss = 0;
        for(int i = 0; i < yDat.columns(); i++){
            DoubleMatrix2D wxTemp = wxProd.viewPart(0, lcumsum[i], n, l[i]);
            //need to copy init values for calculating nll
            DoubleMatrix2D wxTemp0 = wxTemp.copy();

            // does this need to be done in log space??
            wxTemp.assign(Functions.exp);
            DoubleMatrix1D invDenom = factory1D.make(n,1.0).assign(margSum(wxTemp, 2), Functions.div);
            wxTemp.assign(alg.mult(factory2D.diagonal(invDenom), wxTemp));
            for(int k = 0; k < n; k++){
                DoubleMatrix1D curRow = wxTemp.viewRow(k);
                DoubleMatrix1D curRow0 = wxTemp0.viewRow(k);

                catloss -= curRow0.get((int) yDat.get(k, i) - 1);
                catloss += logsumexp(curRow0);


                //wxtemp(sub2ind(size(wxtemp),(1:n)',Y(:,r)))=wxtemp(sub2ind(size(wxtemp),(1:n)',Y(:,r)))-1;
                curRow.set((int) yDat.get(k,i)-1, curRow.get((int) yDat.get(k,i)-1) - 1);
            }
        }

        //gradalpha2=sum(wxprod,1)';
        gradOut.alpha2 = margSum(wxProd,1);

        //gradw=X'*wxprod;
        DoubleMatrix2D gradW = alg.mult(alg.transpose(xDat), wxProd);

        //gradtheta=gradtheta+gradw';
        gradOut.theta.assign(alg.transpose(gradW), Functions.plus);

        //gradphi=D'*wxprod;
        gradOut.phi = alg.mult(alg.transpose(dDat), wxProd);

        //zero out gradphi diagonal
        //for r=1:q
        //gradphi(Lsum(r)+1:Lsum(r+1),Lsum(r)+1:Lsum(r+1))=0;
        //end
        for(int i = 0; i < q; i++){
            gradOut.phi.viewPart(lcumsum[i], lcumsum[i], l[i], l[i]).assign(0);
        }

        //gradphi=tril(gradphi)'+triu(gradphi);
        DoubleMatrix2D lowerPhi = alg.transpose(lowerTri(gradOut.phi.copy(), 0));
        upperTri(gradOut.phi, 0).assign(lowerPhi, Functions.plus);

        /*
        for s=1:p
            gradbetad(s)=-n/(2*betad(s))+1/2*norm(res(:,s))^2-res(:,s)'*(Xbeta(:,s)+Dtheta(:,s));
        end
         */
        gradOut.betad = factory1D.make(xDat.columns());
        for(int i = 0; i < p; i++){
            gradOut.betad.set(i, -n / (2.0 * par.betad.get(i)) + alg.norm2(tempLoss.viewColumn(i)) / 2.0 -
                    alg.mult(tempLoss.viewColumn(i), xBeta.viewColumn(i).copy().assign(dTheta.viewColumn(i), Functions.plus)));
        }
        
        /*
        for (int i = 0; i < q; i++)
        	par.eta.viewRow(lcumsum[i]).assign(0.0);
        */
        /*
         * Cox regression loss, gamma and eta gradients
         */

        double coxloss = 0.0;
        double tempcoxloss;
        double rs_sum;
        double Hsum;
        double Hsumtemp;
        double sub;
        double frac;
        DoubleMatrix1D temp = factory1D.make(n, 0.0);
        DoubleMatrix1D Hsumx;
        DoubleMatrix1D Hsumy;
        DoubleMatrix1D Hsumtempx;
        DoubleMatrix1D Hsumtempy;
        DoubleMatrix1D numx;
        DoubleMatrix1D numy;
        DoubleMatrix1D sub_numx;
        DoubleMatrix1D sub_numy;

        gradOut.gamma = factory2D.make(p,r);
        gradOut.eta = factory2D.make(lsum,r);

        for (int i = 0; i < r; i++) {
//            System.out.println(Thread.currentThread().getName() + ": xDat: " + xDat);
//            System.out.println(Thread.currentThread().getName() + ": gamma: " + par.gamma.viewColumn(i));
            temp.assign(alg.mult(xDat, par.gamma.viewColumn(i)));
//            System.out.println(Thread.currentThread().getName() + ": temp_gamma_only: " + temp);
            temp.assign(alg.mult(dDat, par.eta.viewColumn(i)), Functions.plus);
//            System.out.println(Thread.currentThread().getName() + ": temp_pre_exp: " + temp);
            temp.assign(Functions.exp);

//            System.out.println(Thread.currentThread().getName() + ": temp: " + temp);

            DoubleMatrix1D sTemp = temp.copy();

            for (int j = 0; j < n; j++)
                sTemp.set(j, temp.get(order[j][i]));

            temp.assign(sTemp);

//            System.out.println(Thread.currentThread().getName() + ": sTemp: " + temp);

            tempcoxloss = 0.0;
            rs_sum = temp.zSum();

//            System.out.println(Thread.currentThread().getName() + ": rs_sum: " + rs_sum);

            numx = factory1D.make(p, 0.0);
            numy = factory1D.make(lsum, 0.0);
            for (int j = 0; j < n; j++) {
                numx.assign(xDat.viewRow(order[j][i]).copy().assign(Functions.mult(temp.get(j))), Functions.plus);
                numy.assign(dDat.viewRow(order[j][i]).copy().assign(Functions.mult(temp.get(j))), Functions.plus);
            }

            int j = 0;
            int idx = 0;
            int m;

            while (idx < n) {
//                System.out.println(idx + " : " + H[j][i]);
                Hsum = 0.0;
                Hsumtemp = 0.0;
                Hsumx = factory1D.make(p, 0.0);
                Hsumy = factory1D.make(lsum, 0.0);
                Hsumtempx = factory1D.make(p, 0.0);
                Hsumtempy = factory1D.make(lsum, 0.0);
                sub_numx = factory1D.make(p, 0.0);
                sub_numy = factory1D.make(lsum, 0.0);
                m = 0;
                sub = 0;
                for (int k = 0; k < H[j][i]; k++) {
                    m += cDat.get(order[idx+k][i], i);

                    Hsum += cDat.get(order[idx+k][i], i) * (alg.mult(xDat.viewRow(order[idx+k][i]), par.gamma.viewColumn(i)) + alg.mult(dDat.viewRow(order[idx+k][i]), par.eta.viewColumn(i)));
                    Hsumtemp += cDat.get(order[idx+k][i], i) * temp.get(idx+k);

                    if (cDat.get(order[idx+k][i], i) == 1) {
                        Hsumx.assign(xDat.viewRow(order[idx + k][i]), Functions.plus);
                        Hsumy.assign(dDat.viewRow(order[idx + k][i]), Functions.plus);

                        Hsumtempx.assign(xDat.viewRow(order[idx + k][i]).copy().assign(Functions.mult(temp.get(idx + k))), Functions.plus);
                        Hsumtempy.assign(dDat.viewRow(order[idx + k][i]).copy().assign(Functions.mult(temp.get(idx + k))), Functions.plus);
                    }

                    sub += temp.get(idx+k);

                    sub_numx.assign(xDat.viewRow(order[idx + k][i]).copy().assign(Functions.mult(temp.get(idx + k))), Functions.plus);
                    sub_numy.assign(dDat.viewRow(order[idx + k][i]).copy().assign(Functions.mult(temp.get(idx + k))), Functions.plus);
                }

                if ((sub > rs_sum) || (rs_sum <= 0)){
                    rs_sum = temp.viewPart(idx, n-idx).zSum();
                }

                assert Hsumtemp <= sub : "Hsumtemp > sub";
                assert sub <= rs_sum : "sub > rs_sum";
                assert idx < n : "idx >= n";

                tempcoxloss -= Hsum;

                gradOut.gamma.viewColumn(i).assign(Hsumx, Functions.minus);
                gradOut.eta.viewColumn(i).assign(Hsumy, Functions.minus);
                for (int l = 0; l < m; l++) {
                    frac = l / ((double) m);
                    tempcoxloss += Math.log(rs_sum - frac * Hsumtemp);

                    gradOut.gamma.viewColumn(i).assign(numx.copy().assign(Hsumtempx.copy().assign(Functions.mult(frac)), Functions.minus).assign(Functions.div(rs_sum - frac * Hsumtemp)), Functions.plus);
                    gradOut.eta.viewColumn(i).assign(numy.copy().assign(Hsumtempy.copy().assign(Functions.mult(frac)), Functions.minus).assign(Functions.div(rs_sum - frac * Hsumtemp)), Functions.plus);
                }

                idx += H[j][i];
                j++;
                rs_sum -= sub;
                numx.assign(sub_numx, Functions.minus);
                numy.assign(sub_numy, Functions.minus);
            }
            coxloss += tempcoxloss / (double) Ne[i]; // / cDat.zSum();

            gradOut.gamma.viewColumn(i).assign(Functions.div(Ne[i]));
            gradOut.eta.viewColumn(i).assign(Functions.div(Ne[i]));
        }

        gradOut.alpha1.assign(Functions.div((double) n));
        gradOut.alpha2.assign(Functions.div((double) n));
        gradOut.betad.assign(Functions.div((double) n));
        gradOut.beta.assign(Functions.div((double) n));
        gradOut.theta.assign(Functions.div((double) n));
        gradOut.phi.assign(Functions.div((double) n));
//        gradOut.gamma.assign(Functions.div((double) n));
//        gradOut.eta.assign(Functions.div((double) n));

//        System.out.println(Thread.currentThread().getName() + ": dgamma: \n" + gradOut.gamma.viewDice());
//        System.out.println(Thread.currentThread().getName() + ": deta: \n" + gradOut.eta.viewDice());
        
//        for (int i = 0; i < q; i++)
//        	gradOut.eta.viewRow(lcumsum[i]).assign(0.0);

        gradOutVec.assign(gradOut.toMatrix1D());
        return (sqloss + catloss)/((double) n)  + coxloss;
    }

    /**
     * Calculates penalty term of objective function
     *
     * @param parIn
     * @return
     */
    public double nonSmoothValue(DoubleMatrix1D parIn){
        //DoubleMatrix1D tlam = lambda.copy().assign(Functions.mult(t));
        //Dimension checked in constructor
        //par is a copy so we can update it
    	survivalMGMParams par = new survivalMGMParams(parIn, p, lsum, r);

//        calcCensoredWeights(par);

        //penbeta = t(1).*(wv(1:p)'*wv(1:p));
        //betascale=zeros(size(beta));
        //betascale=max(0,1-penbeta./abs(beta));
        DoubleMatrix2D weightMat = alg.multOuter(weights,
                weights, null);

//        weightMat.viewPart(0, p+q, p+q, r) .assign(censoredWeights.copy());

        //int p = xDat.columns();

        //weight beta
        //betaw = (wv(1:p)'*wv(1:p)).*abs(beta);
        //betanorms=sum(betaw(:));
        DoubleMatrix2D betaWeight = weightMat.viewPart(0, 0, p, p);
        DoubleMatrix2D absBeta = par.beta.copy().assign(Functions.abs);
        double betaNorms = absBeta.assign(betaWeight, Functions.mult).zSum();


        /*
        thetanorms=0;
        for s=1:p
            for j=1:q
                tempvec=theta(Lsums(j)+1:Lsums(j+1),s);
                thetanorms=thetanorms+(wv(s)*wv(p+j))*norm(tempvec);
            end
        end
        */
        double thetaNorms = 0;
        for(int i = 0; i < p; i++){
            for(int j = 0; j < q; j++){
                DoubleMatrix1D tempVec = par.theta.viewColumn(i).viewPart(lcumsum[j], l[j]);
                thetaNorms += weightMat.get(i, p+j)*Math.sqrt(alg.norm2(tempVec));
            }
        }

        /*
        for r=1:q
            for j=1:q
                if r<j
                    tempmat=phi(Lsums(r)+1:Lsums(r+1),Lsums(j)+1:Lsums(j+1));
                    tempmat=max(0,1-t(3)*(wv(p+r)*wv(p+j))/norm(tempmat))*tempmat; % Lj by 2*Lr
                    phinorms=phinorms+(wv(p+r)*wv(p+j))*norm(tempmat,'fro');
                    phi( Lsums(r)+1:Lsums(r+1),Lsums(j)+1:Lsums(j+1) )=tempmat;
                end
            end
        end
         */
        double phiNorms = 0;
        for(int i = 0; i < q; i++){
            for(int j = i+1; j < q; j++){
                DoubleMatrix2D tempMat = par.phi.viewPart(lcumsum[i], lcumsum[j], l[i], l[j]);
                phiNorms += weightMat.get(p+i,p+j)*alg.normF(tempMat);
            }
        }

//        DoubleMatrix2D gammaWeight = censoredWeights.viewPart(0, 0, p, r);
        DoubleMatrix2D gammaWeight = factory2D.make(p,r,0);
        for (int i = 0; i < r; i++) gammaWeight.viewColumn(i).assign(weights.viewPart(0, p));
//                DoubleMatrix1D gammaWeight = weights.viewPart(0, p);
        DoubleMatrix2D absgamma = par.gamma.copy().assign(Functions.abs);
        double gammaNorms = absgamma.assign(gammaWeight, Functions.mult).zSum();
        /*
        for (int i = 0; i < r; i++) {
        	gammaNorms += alg.mult(par.gamma.viewColumn(i).copy().assign(Functions.abs), weightMat.viewPart();
        }
        /*
        DoubleMatrix1D weightsByCat = factory1D.make(lsum);
        for (int i = 0; i < q; i++) {
        	par.eta.viewRow(lcumsum[i]).assign(0.0);
        	for (int j = 0; j < q; j++) {
        		weightsByCat.viewPart(lcumsum[j], l[j]).assign(weights.get(p+j)); // /(l[j]-1)
        		// weightsByCat.viewPart(lcumsum[j], l[j]).assign(1.0);
        	}
        }
        */
        double etaNorms = 0;
        for (int i = 0; i < r; i++) {
            /*
            for(int j = 0; j < q; j++){
                DoubleMatrix1D tempVec = par.eta.viewColumn(i).viewPart(lcumsum[j], l[j]).copy();
                etaNorms += weights.get(p+j) * tempVec.assign(Functions.abs).zSum();
            }
            */
//        	etaNorms += alg.mult(par.eta.viewColumn(i).copy().assign(Functions.abs), weightsByCat);
        	for(int j = 0; j < q; j++){
                DoubleMatrix1D tempVec = par.eta.viewColumn(i).viewPart(lcumsum[j], l[j]);
//                etaNorms += censoredWeights.get(p+j, i)  * Math.sqrt(alg.norm2(tempVec));
//                etaNorms += weights.get(p+j)  * Math.sqrt(alg.norm2(tempVec));
                etaNorms += Math.sqrt(alg.norm2(tempVec));
            }
        }

        return lambda.get(0)*betaNorms + lambda.get(1)*thetaNorms + lambda.get(2)*phiNorms + lambda.get(3)*gammaNorms + lambda.get(4)*etaNorms;
    }


    /**
     * Gradient of the pseudolikelihood
     *
     * @param parIn
     * @return
     */
    public DoubleMatrix1D smoothGradient(DoubleMatrix1D parIn){
        int n = xDat.rows();
        survivalMGMParams grad = new survivalMGMParams();

        //
        survivalMGMParams par = new survivalMGMParams(parIn, p, lsum, r);
        upperTri(par.beta, 1);
        par.beta.assign(alg.transpose(par.beta), Functions.plus);

        for(int i = 0; i < q; i++){
            par.phi.viewPart(lcumsum[i], lcumsum[i], l[i], l[i]).assign(0);
        }
        upperTri(par.phi, 0);
        par.phi.assign(alg.transpose(par.phi), Functions.plus);

        //Xbeta=X*beta*diag(1./betad);
        //Dtheta=D*theta*diag(1./betad);
        DoubleMatrix2D divBetaD = factory2D.diagonal(factory1D.make(p, 1.0).assign(par.betad, Functions.div));

        DoubleMatrix2D xBeta = alg.mult(alg.mult(xDat, par.beta), divBetaD);
        DoubleMatrix2D dTheta = alg.mult(alg.mult(dDat, par.theta), divBetaD);

        //res=Xbeta-X+e*alpha1'+Dtheta;
        DoubleMatrix2D negLoss = factory2D.make(n, xDat.columns());

        //wxprod=X*(theta')+D*phi+e*alpha2';
        DoubleMatrix2D wxProd = alg.mult(xDat, alg.transpose(par.theta));
        wxProd.assign(alg.mult(dDat, par.phi), Functions.plus);
        for(int i = 0; i < n; i++){
            for(int j = 0; j < p; j++){
                negLoss.set(i,j, xBeta.get(i,j) - xDat.get(i,j) + par.alpha1.get(j) + dTheta.get(i,j));
            }
            for(int j = 0; j < dDat.columns(); j++){
                wxProd.set(i,j,wxProd.get(i,j) + par.alpha2.get(j));
            }
        }

        //gradbeta=X'*(res);
        grad.beta = alg.mult(alg.transpose(xDat), negLoss);

        //gradbeta=gradbeta-diag(diag(gradbeta)); % zero out diag
        //gradbeta=tril(gradbeta)'+triu(gradbeta);
        DoubleMatrix2D lowerBeta = alg.transpose(lowerTri(grad.beta.copy(), -1));
        upperTri(grad.beta, 1).assign(lowerBeta, Functions.plus);

        //gradalpha1=diag(betad)*sum(res,1)';
        grad.alpha1 = alg.mult(factory2D.diagonal(par.betad),margSum(negLoss, 1));

        //gradtheta=D'*(res);
        grad.theta = alg.mult(alg.transpose(dDat), negLoss);

        /*
        wxprod=X*(theta')+D*phi+e*alpha2'; %this is n by Ltot
        Lsum=[0;cumsum(L)];
        for r=1:q
            idx=Lsum(r)+1:Lsum(r)+L(r);
            wxtemp=wxprod(:,idx); %n by L(r)
            denom=sum(exp(wxtemp),2); % this is n by 1
            wxtemp=diag(sparse(1./denom))*exp(wxtemp);
            wxtemp(sub2ind(size(wxtemp),(1:n)',Y(:,r)))=wxtemp(sub2ind(size(wxtemp),(1:n)',Y(:,r)))-1;
            wxprod(:,idx)=wxtemp;
        end
        */

        for(int i = 0; i < yDat.columns(); i++){
            DoubleMatrix2D wxTemp = wxProd.viewPart(0, lcumsum[i], n, l[i]);

            // does this need to be done in log space??
            wxTemp.assign(Functions.exp);
            DoubleMatrix1D invDenom = factory1D.make(n,1.0).assign(margSum(wxTemp, 2), Functions.div);
            wxTemp.assign(alg.mult(factory2D.diagonal(invDenom), wxTemp));
            for(int k = 0; k < n; k++){
                DoubleMatrix1D curRow = wxTemp.viewRow(k);
                //wxtemp(sub2ind(size(wxtemp),(1:n)',Y(:,r)))=wxtemp(sub2ind(size(wxtemp),(1:n)',Y(:,r)))-1;
                curRow.set((int) yDat.get(k,i)-1, curRow.get((int) yDat.get(k,i)-1) - 1);
            }
        }

        //gradalpha2=sum(wxprod,1)';
        grad.alpha2 = margSum(wxProd,1);

        //gradw=X'*wxprod;
        DoubleMatrix2D gradW = alg.mult(alg.transpose(xDat), wxProd);

        //gradtheta=gradtheta+gradw';
        grad.theta.assign(alg.transpose(gradW), Functions.plus);

        //gradphi=D'*wxprod;
        grad.phi = alg.mult(alg.transpose(dDat), wxProd);

        //zero out gradphi diagonal
        //for r=1:q
        //gradphi(Lsum(r)+1:Lsum(r+1),Lsum(r)+1:Lsum(r+1))=0;
        //end
        for(int i = 0; i < q; i++){
            grad.phi.viewPart(lcumsum[i], lcumsum[i], l[i], l[i]).assign(0);
        }

        //gradphi=tril(gradphi)'+triu(gradphi);
        DoubleMatrix2D lowerPhi = alg.transpose(lowerTri(grad.phi.copy(), 0));
        upperTri(grad.phi, 0).assign(lowerPhi, Functions.plus);

        /*
        for s=1:p
            gradbetad(s)=-n/(2*betad(s))+1/2*norm(res(:,s))^2-res(:,s)'*(Xbeta(:,s)+Dtheta(:,s));
        end
         */
        grad.betad = factory1D.make(xDat.columns());
        for(int i = 0; i < p; i++){
            grad.betad.set(i, -n / (2.0 * par.betad.get(i)) + alg.norm2(negLoss.viewColumn(i)) / 2.0 -
                    alg.mult(negLoss.viewColumn(i), xBeta.viewColumn(i).copy().assign(dTheta.viewColumn(i), Functions.plus)));
        }
        /*
        for (int i = 0; i < q; i++)
        	par.eta.viewRow(lcumsum[i]).assign(0.0);
        */
        /*
         * Cox regression loss, gamma and eta gradients
         */

        double coxloss = 0.0;
        double tempcoxloss;
        double rs_sum;
        double Hsum;
        double Hsumtemp;
        double sub;
        double frac;
        DoubleMatrix1D temp = factory1D.make(n, 0.0);
        DoubleMatrix1D Hsumx;
        DoubleMatrix1D Hsumy;
        DoubleMatrix1D Hsumtempx;
        DoubleMatrix1D Hsumtempy;
        DoubleMatrix1D numx;
        DoubleMatrix1D numy;
        DoubleMatrix1D sub_numx;
        DoubleMatrix1D sub_numy;

        grad.gamma = factory2D.make(p,r);
        grad.eta = factory2D.make(lsum,r);

        for (int i = 0; i < r; i++) {
//            System.out.println(Thread.currentThread().getName() + ": xDat: " + xDat);
//            System.out.println(Thread.currentThread().getName() + ": gamma: " + par.gamma.viewColumn(i));
            temp.assign(alg.mult(xDat, par.gamma.viewColumn(i)));
//            System.out.println(Thread.currentThread().getName() + ": temp_gamma_only: " + temp);
            temp.assign(alg.mult(dDat, par.eta.viewColumn(i)), Functions.plus);
//            System.out.println(Thread.currentThread().getName() + ": temp_pre_exp: " + temp);
            temp.assign(Functions.exp);

//            System.out.println(Thread.currentThread().getName() + ": temp: " + temp);

            DoubleMatrix1D sTemp = temp.copy();

            for (int j = 0; j < n; j++)
                sTemp.set(j, temp.get(order[j][i]));

            temp.assign(sTemp);

//            System.out.println(Thread.currentThread().getName() + ": sTemp: " + temp);

            tempcoxloss = 0.0;
            rs_sum = temp.zSum();

//            System.out.println(Thread.currentThread().getName() + ": rs_sum: " + rs_sum);

            numx = factory1D.make(p, 0.0);
            numy = factory1D.make(lsum, 0.0);
            for (int j = 0; j < n; j++) {
                numx.assign(xDat.viewRow(order[j][i]).copy().assign(Functions.mult(temp.get(j))), Functions.plus);
                numy.assign(dDat.viewRow(order[j][i]).copy().assign(Functions.mult(temp.get(j))), Functions.plus);
            }

            int j = 0;
            int idx = 0;
            int m;

            while (idx < n) {
//                System.out.println(idx + " : " + H[j][i]);
                Hsum = 0.0;
                Hsumtemp = 0.0;
                Hsumx = factory1D.make(p, 0.0);
                Hsumy = factory1D.make(lsum, 0.0);
                Hsumtempx = factory1D.make(p, 0.0);
                Hsumtempy = factory1D.make(lsum, 0.0);
                sub_numx = factory1D.make(p, 0.0);
                sub_numy = factory1D.make(lsum, 0.0);
                m = 0;
                sub = 0;
                for (int k = 0; k < H[j][i]; k++) {
                    m += cDat.get(order[idx+k][i], i);

//                    Hsum += cDat.get(order[idx+k][i], i) * (alg.mult(xDat.viewRow(order[idx+k][i]), par.gamma.viewColumn(i)) + alg.mult(dDat.viewRow(order[idx+k][i]), par.eta.viewColumn(i)));
                    Hsumtemp += cDat.get(order[idx+k][i], i) * temp.get(idx+k);

                    if (cDat.get(order[idx+k][i], i) == 1) {
                        Hsumx.assign(xDat.viewRow(order[idx + k][i]), Functions.plus);
                        Hsumy.assign(dDat.viewRow(order[idx + k][i]), Functions.plus);

                        Hsumtempx.assign(xDat.viewRow(order[idx + k][i]).copy().assign(Functions.mult(temp.get(idx + k))), Functions.plus);
                        Hsumtempy.assign(dDat.viewRow(order[idx + k][i]).copy().assign(Functions.mult(temp.get(idx + k))), Functions.plus);
                    }

                    sub += temp.get(idx+k);

                    sub_numx.assign(xDat.viewRow(order[idx + k][i]).copy().assign(Functions.mult(temp.get(idx + k))), Functions.plus);
                    sub_numy.assign(dDat.viewRow(order[idx + k][i]).copy().assign(Functions.mult(temp.get(idx + k))), Functions.plus);
                }

                if ((sub > rs_sum) || (rs_sum <= 0)){
                    rs_sum = temp.viewPart(idx, n-idx).zSum();
                }

                assert Hsumtemp <= sub : "Hsumtemp > sub";
                assert sub <= rs_sum : "sub > rs_sum";
                assert idx < n : "idx >= n";

//                tempcoxloss -= Hsum;

                grad.gamma.viewColumn(i).assign(Hsumx, Functions.minus);
                grad.eta.viewColumn(i).assign(Hsumy, Functions.minus);
                for (int l = 0; l < m; l++) {
                    frac = l / ((double) m);
//                    tempcoxloss += Math.log(rs_sum - frac * Hsumtemp);

                    grad.gamma.viewColumn(i).assign(numx.copy().assign(Hsumtempx.copy().assign(Functions.mult(frac)), Functions.minus).assign(Functions.div(rs_sum - frac * Hsumtemp)), Functions.plus);
                    grad.eta.viewColumn(i).assign(numy.copy().assign(Hsumtempy.copy().assign(Functions.mult(frac)), Functions.minus).assign(Functions.div(rs_sum - frac * Hsumtemp)), Functions.plus);
                }

                idx += H[j][i];
                j++;
                rs_sum -= sub;
                numx.assign(sub_numx, Functions.minus);
                numy.assign(sub_numy, Functions.minus);
            }
//            coxloss += tempcoxloss;

            grad.gamma.viewColumn(i).assign(Functions.div(Ne[i]));
            grad.eta.viewColumn(i).assign(Functions.div(Ne[i]));
        }

        grad.alpha1.assign(Functions.div((double) n));
        grad.alpha2.assign(Functions.div((double) n));
        grad.betad.assign(Functions.div((double) n));
        grad.beta.assign(Functions.div((double) n));
        grad.theta.assign(Functions.div((double) n));
        grad.phi.assign(Functions.div((double) n));
//        grad.gamma.assign(Functions.div((double) n));
//        grad.eta.assign(Functions.div((double) n));
        
//        for (int i = 0; i < q; i++)
//        	grad.eta.viewRow(lcumsum[i]).assign(0.0);
        
//        System.out.println(grad.theta);
//        System.out.println(grad.gamma.viewDice());
//        System.out.println(grad.eta.viewDice());

//        System.out.println(Thread.currentThread().getName() + ": dgamma: \n" + grad.gamma.viewDice());
//        System.out.println(Thread.currentThread().getName() + ": deta: \n" + grad.eta.viewDice());

        return grad.toMatrix1D();
    }

    /**
     * A proximal operator for the MGM
     *
     * @param t parameter for operator, must be positive
     * @param X input vector to operator
     * @return output vector, same dimension as X
     */
    public DoubleMatrix1D proximalOperator(double t, DoubleMatrix1D X) {
            //System.out.println("PROX with t = " + t);
        if(t <= 0)
            throw new IllegalArgumentException("t must be positive: " + t);


        DoubleMatrix1D tlam = lambda.copy().assign(Functions.mult(t));

        //Constructor copies and checks dimension
        //par is a copy so we can update it
        survivalMGMParams par = new survivalMGMParams(X.copy(), p, lsum, r);

//        calcCensoredWeights(par);

        //penbeta = t(1).*(wv(1:p)'*wv(1:p));
        //betascale=zeros(size(beta));
        //betascale=max(0,1-penbeta./abs(beta));
        DoubleMatrix2D weightMat = alg.multOuter(weights,
                weights, null);

//        weightMat.viewPart(0, p+q, p+q, r) .assign(censoredWeights.copy());

        DoubleMatrix2D betaWeight = weightMat.viewPart(0, 0, p, p);
        DoubleMatrix2D betascale = betaWeight.copy().assign(Functions.mult(-tlam.get(0)));
        betascale.assign(par.beta.copy().assign(Functions.abs), Functions.div);
        betascale.assign(Functions.plus(1));
        betascale.assign(Functions.max(0));

        //beta=beta.*betascale;
        //par.beta.assign(betascale, Functions.mult);
        for(int i= 0; i < p; i++){
            for(int j = 0; j < p; j++){
                double curVal =  par.beta.get(i,j);
                if(curVal !=0){
                    par.beta.set(i,j, curVal*betascale.get(i,j));
                }
            }
        }

        //weight beta
        //betaw = (wv(1:p)'*wv(1:p)).*beta;
        //betanorms=sum(abs(betaw(:)));
        //double betaNorm = betaWeight.copy().assign(par.beta, Functions.mult).assign(Functions.abs).zSum();

        /*
        thetanorms=0;
        for s=1:p
            for j=1:q
                tempvec=theta(Lsums(j)+1:Lsums(j+1),s);
                tempvec=max(0,1-t(2)*(wv(s)*wv(p+j))/norm(tempvec))*tempvec;
                thetanorms=thetanorms+(wv(s)*wv(p+j))*norm(tempvec);
                theta(Lsums(j)+1:Lsums(j+1),s)=tempvec(1:L(j));
            end
        end
        */
        for(int i = 0; i < p; i++){
            for(int j = 0; j < q; j++){
                DoubleMatrix1D tempVec = par.theta.viewColumn(i).viewPart(lcumsum[j], l[j]);
                //double thetaScale = Math.max(0, 1 - tlam.get(1)*weightMat.get(i, p+j)/Math.sqrt(alg.norm2(tempVec)));
                // double foo = norm2(tempVec);
                double thetaScale = Math.max(0, 1 - tlam.get(1) * weightMat.get(i, p+j)/norm2(tempVec));
                tempVec.assign(Functions.mult(thetaScale));
            }
        }

        /*
        for r=1:q
            for j=1:q
                if r<j
                    tempmat=phi(Lsums(r)+1:Lsums(r+1),Lsums(j)+1:Lsums(j+1));
                    tempmat=max(0,1-t(3)*(wv(p+r)*wv(p+j))/norm(tempmat))*tempmat; % Lj by 2*Lr
                    phinorms=phinorms+(wv(p+r)*wv(p+j))*norm(tempmat,'fro');
                    phi( Lsums(r)+1:Lsums(r+1),Lsums(j)+1:Lsums(j+1) )=tempmat;
                end
            end
        end
         */
        for(int i = 0; i < q; i++){
            for(int j = i+1; j < q; j++){
                DoubleMatrix2D tempMat = par.phi.viewPart(lcumsum[i], lcumsum[j], l[i], l[j]);

                //Not sure why this isnt Frobenius norm...
                //double phiScale = Math.max(0, 1-tlam.get(2)*weightMat.get(p+i,p+j)/alg.norm2(tempMat));
                //double phiScale = Math.max(0, 1 - tlam.get(2) * weightMat.get(p + i,p+j)/norm2(tempMat));
                double phiScale = Math.max(0, 1-tlam.get(2)*weightMat.get(p+i,p+j)/alg.normF(tempMat));
                tempMat.assign(Functions.mult(phiScale));
            }
        }

//        DoubleMatrix2D gammaWeight = censoredWeights.viewPart(0, 0, p, r);
        DoubleMatrix2D gammaWeight = factory2D.make(p,r,0);
        for (int i = 0; i < r; i++) gammaWeight.viewColumn(i).assign(weights.viewPart(0, p));
        DoubleMatrix2D gammascale = gammaWeight.copy().assign(Functions.mult(-tlam.get(3)));
        DoubleMatrix2D absgamma = par.gamma.copy().assign(Functions.abs);
        gammascale.assign(absgamma, Functions.div);
        gammascale.assign(Functions.plus(1));
        gammascale.assign(Functions.max(0));
        par.gamma.assign(gammascale, Functions.mult);

//        double absgamma;
//        double gammascale;
////        double gammaNorms = 0;
//        for (int i = 0; i < r; i++) {
//            for (int j = 0; j < p; j++) {
//                absgamma = Math.abs(par.gamma.get(j,i));
//                gammascale = Math.max(0, 1 - censoredWeights.get(j, i) * tlam.get(3) / absgamma);
//                if (absgamma == 0) gammascale = 0.0;
//                par.gamma.set(j,i, gammascale * par.gamma.get(j,i));
////                gammaNorms += censoredWeights.get(j, i) * absgamma;
//            }
//        }

        /*
        for (int i = 0; i < r; i++) {
            DoubleMatrix1D gammascale = factory1D.make(p, -tlam.get(3));
//        	DoubleMatrix1D gammascale = weights.viewPart(0,p).copy().assign(Functions.mult(-tlam.get(3)));
        	DoubleMatrix1D absgamma = par.gamma.viewColumn(i).copy().assign(Functions.abs);
        	gammascale.assign(absgamma, Functions.div);
        	gammascale.assign(Functions.plus(1));
            gammascale.assign(Functions.max(0));
            par.gamma.viewColumn(i).assign(gammascale, Functions.mult);
        }

        /*
        DoubleMatrix1D weightsByCat = factory1D.make(lsum);
        for (int i = 0; i < q; i++) {
            par.eta.viewRow(lcumsum[i]).assign(0.0);
            for (int j = 0; j < q; j++) {
                weightsByCat.viewPart(lcumsum[j], l[j]).assign(weights.get(p+j));
            }
        }
        */
        for (int i = 0; i < r; i++) {
            /*
            DoubleMatrix1D etascale = weightsByCat.copy().assign(Functions.mult(-tlam.get(4)));
            DoubleMatrix1D abseta = par.eta.viewColumn(i).copy().assign(Functions.abs);
            etascale.assign(abseta, Functions.div);
            etascale.assign(Functions.plus(1));
            etascale.assign(Functions.max(0));
            par.eta.viewColumn(i).assign(etascale, Functions.mult);
            */

        	for(int j = 0; j < q; j++){
//        	    int minIdx = 0;
//        	    double min = Double.POSITIVE_INFINITY;
                DoubleMatrix1D tempVec = par.eta.viewColumn(i).viewPart(lcumsum[j], l[j]);
//                for (int k = 0; k < l[j]; k++) {
//                    if (Math.abs(tempVec.get(k)) < min) {
//                        minIdx = k;
//                    }
//                }
//                tempVec.viewPart(minIdx, 1).assign(0.0);
//                double etaScale;
//                if (norm2(tempVec)==0) {
//                    etaScale = 0.0;
//                } else {
//                    etaScale = Math.max(0, 1 - censoredWeights.get(p + j, i) * tlam.get(4) / norm2(tempVec));
//                }
//                tempVec.assign(Functions.mult(etaScale));
//                tempVec.assign(Functions.mult(Math.max(0, 1 - weights.get(p + j) * tlam.get(4) / norm2(tempVec))));
                tempVec.assign(Functions.mult(Math.max(0, 1 - tlam.get(4) / norm2(tempVec))));
            }
        }
        
        return par.toMatrix1D();
    }

    /**
     * Calculates penalty term and proximal operator at the same time for speed
     *
     * @param t proximal operator parameter
     * @param X input
     * @param pX prox operator solution
     * @return value of penalty term
     */
    public double nonSmooth(double t, DoubleMatrix1D X, DoubleMatrix1D pX) {

        //System.out.println("PROX with t = " + t);
        // double nonSmooth = 0;

        DoubleMatrix1D tlam = lambda.copy().assign(Functions.mult(t));

        //Constructor copies and checks dimension
        //par is a copy so we can update it
        survivalMGMParams par = new survivalMGMParams(X, p, lsum, r);

//        calcCensoredWeights(par);
        
        // System.out.println(par.toString());

        //penbeta = t(1).*(wv(1:p)'*wv(1:p));
        //betascale=zeros(size(beta));
        //betascale=max(0,1-penbeta./abs(beta));
        DoubleMatrix2D weightMat = alg.multOuter(weights,
                weights, null);

//        weightMat.viewPart(0, p+q, p+q, r) .assign(censoredWeights.copy());

        DoubleMatrix2D betaWeight = weightMat.viewPart(0, 0, p, p); // this is constant
        DoubleMatrix2D betascale = betaWeight.copy().assign(Functions.mult(-tlam.get(0)));
        DoubleMatrix2D absBeta = par.beta.copy().assign(Functions.abs);
        betascale.assign(absBeta, Functions.div);
        betascale.assign(Functions.plus(1));
        betascale.assign(Functions.max(0));


        double betaNorms  = 0;

        //beta=beta.*betascale;
        //par.beta.assign(betascale, Functions.mult);

        for(int i= 0; i < p; i++){
            for(int j = 0; j < p; j++){
                double curVal =  par.beta.get(i,j);
                if(curVal !=0){
                    curVal=curVal * betascale.get(i,j);
                    par.beta.set(i,j,curVal);
                    betaNorms += Math.abs(betaWeight.get(i,j)*curVal);
                }
            }
        }

        //weight beta
        //betaw = (wv(1:p)'*wv(1:p)).*beta;
        //betanorms=sum(abs(betaw(:)));
        //double betaNorm = betaWeight.copy().assign(par.beta, Functions.mult).assign(Functions.abs).zSum();

        /*
        thetanorms=0;
        for s=1:p
            for j=1:q
                tempvec=theta(Lsums(j)+1:Lsums(j+1),s);
                tempvec=max(0,1-t(2)*(wv(s)*wv(p+j))/norm(tempvec))*tempvec;
                thetanorms=thetanorms+(wv(s)*wv(p+j))*norm(tempvec);
                theta(Lsums(j)+1:Lsums(j+1),s)=tempvec(1:L(j));
            end
        end
        */
        double thetaNorms = 0;
        for(int i = 0; i < p; i++){
            for(int j = 0; j < q; j++){
                DoubleMatrix1D tempVec = par.theta.viewColumn(i).viewPart(lcumsum[j], l[j]);
                //double thetaScale = Math.max(0, 1 - tlam.get(1)*weightMat.get(i, p+j)/Math.sqrt(alg.norm2(tempVec)));
                //double foo = norm2(tempVec);
                double thetaScale = Math.max(0, 1 - tlam.get(1) * weightMat.get(i, p+j)/norm2(tempVec));
                tempVec.assign(Functions.mult(thetaScale));
                thetaNorms += weightMat.get(i, p+j)*Math.sqrt(alg.norm2(tempVec));
            }
        }

        /*
        for r=1:q
            for j=1:q
                if r<j
                    tempmat=phi(Lsums(r)+1:Lsums(r+1),Lsums(j)+1:Lsums(j+1));
                    tempmat=max(0,1-t(3)*(wv(p+r)*wv(p+j))/norm(tempmat))*tempmat; % Lj by 2*Lr
                    phinorms=phinorms+(wv(p+r)*wv(p+j))*norm(tempmat,'fro');
                    phi( Lsums(r)+1:Lsums(r+1),Lsums(j)+1:Lsums(j+1) )=tempmat;
                end
            end
        end
         */
        double phiNorms = 0;
        for(int i = 0; i < q; i++){
            for(int j = i+1; j < q; j++){
                DoubleMatrix2D tempMat = par.phi.viewPart(lcumsum[i], lcumsum[j], l[i], l[j]);

                //not sure why this isnt Frobenius norm...
                //double phiScale = Math.max(0, 1-tlam.get(2)*weightMat.get(p+i,p+j)/alg.norm2(tempMat));
                //double phiScale = Math.max(0, 1 - tlam.get(2) * weightMat.get(p + i,p+j)/norm2(tempMat));
                double phiScale = Math.max(0, 1-tlam.get(2)*weightMat.get(p+i,p+j)/alg.normF(tempMat));
                tempMat.assign(Functions.mult(phiScale));
                phiNorms += weightMat.get(p+i,p+j)*alg.normF(tempMat);
            }
        }

//        DoubleMatrix2D gammaWeight = censoredWeights.viewPart(0, 0, p, r);
//        DoubleMatrix2D gammascale = gammaWeight.copy().assign(Functions.mult(-tlam.get(3)));
//        DoubleMatrix2D absgamma = par.gamma.copy().assign(Functions.abs);
//        gammascale.assign(absgamma, Functions.div);
//        gammascale.assign(Functions.plus(1));
//        gammascale.assign(Functions.max(0));
        DoubleMatrix2D gammaWeight = factory2D.make(p,r,0);
        for (int i = 0; i < r; i++) gammaWeight.viewColumn(i).assign(weights.viewPart(0, p));
        DoubleMatrix2D gammascale = gammaWeight.copy().assign(Functions.mult(-tlam.get(3)));
        DoubleMatrix2D absgamma = par.gamma.copy().assign(Functions.abs);
        gammascale.assign(absgamma, Functions.div);
        gammascale.assign(Functions.plus(1));
        gammascale.assign(Functions.max(0));
        par.gamma.assign(gammascale, Functions.mult);
        double gammaNorms = absgamma.assign(gammaWeight, Functions.mult).zSum();

//        double absgamma;
//        double gammascale;
//        double gammaNorms = 0;
//        for (int i = 0; i < r; i++) {
//            for (int j = 0; j < p; j++) {
//                absgamma = Math.abs(par.gamma.get(j,i));
//                gammascale = Math.max(0, 1 - censoredWeights.get(j, i) * tlam.get(3) / absgamma);
////                gammascale = Math.max(0, 1 - weights.get(j) * tlam.get(3) / absgamma);
//                if (absgamma == 0) gammascale = 0.0;
//                par.gamma.set(j,i, gammascale * par.gamma.get(j,i));
//                gammaNorms += censoredWeights.get(j, i) * absgamma;
////                gammaNorms += weights.get(j) * absgamma;
//            }
//        }

        /*
        double gammaNorms = 0;
        for (int i = 0; i < r; i++) {
            DoubleMatrix1D gammascale = factory1D.make(p, -tlam.get(3));
//        	DoubleMatrix1D gammascale = weights.viewPart(0,p).copy().assign(Functions.mult(-tlam.get(3)));
        	DoubleMatrix1D absgamma = par.gamma.viewColumn(i).copy().assign(Functions.abs);
        	gammascale.assign(absgamma, Functions.div);
        	gammascale.assign(Functions.plus(1));
            gammascale.assign(Functions.max(0));
            par.gamma.viewColumn(i).assign(gammascale, Functions.mult);
        	gammaNorms += alg.mult(par.gamma.viewColumn(i).copy().assign(Functions.abs), weights.viewPart(0,p));
        }
        */
        double etaNorms = 0;
        /*
        DoubleMatrix1D weightsByCat = factory1D.make(lsum);

        for (int i = 0; i < q; i++) {
            par.eta.viewRow(lcumsum[i]).assign(0.0);
            for (int j = 0; j < q; j++) {
                weightsByCat.viewPart(lcumsum[j], l[j]).assign(weights.get(p+j));
            }
        }
        */

        for (int i = 0; i < r; i++) {
            /*
            DoubleMatrix1D etascale = weightsByCat.copy().assign(Functions.mult(-tlam.get(4)));
            DoubleMatrix1D abseta = par.eta.viewColumn(i).copy().assign(Functions.abs);
//            for(int j = 0; j < q; j++){
//                int minIdx = 0;
//                double min = Double.POSITIVE_INFINITY;
//                DoubleMatrix1D tempVec = abseta.viewPart(lcumsum[j], l[j]);
//                for (int k = 0; k < l[j]; k++) {
//                    if (Math.abs(tempVec.get(k)) < min) {
//                        minIdx = k;
//                    }
//                }
//                tempVec.viewPart(minIdx, 1).assign(0.0);
//            }
        	etascale.assign(abseta, Functions.div);
        	etascale.assign(Functions.plus(1));
        	etascale.assign(Functions.max(0));
            par.eta.viewColumn(i).assign(etascale, Functions.mult);
            etaNorms += alg.mult(par.eta.viewColumn(i).copy().assign(Functions.abs), weights.viewPart(p, q));
             */
            for(int j = 0; j < q; j++){
//                int minIdx = 0;
//                double min = Double.POSITIVE_INFINITY;
                DoubleMatrix1D tempVec = par.eta.viewColumn(i).viewPart(lcumsum[j], l[j]);
//                for (int k = 0; k < l[j]; k++) {
//                    if (Math.abs(tempVec.get(k)) < min) {
//                        minIdx = k;
//                    }
//                }
//                tempVec.viewPart(minIdx, 1).assign(0.0);
//                double etaScale;
//                if (norm2(tempVec)==0) {
//                    etaScale = 0.0;
//                } else {
//                    etaScale = Math.max(0, 1 - censoredWeights.get(p + j, i) * tlam.get(4) / norm2(tempVec));
//                }
//                tempVec.assign(Functions.mult(etaScale));
//                tempVec.assign(Functions.mult(Math.max(0, 1 - weights.get(p + j) * tlam.get(4) / norm2(tempVec))));
                tempVec.assign(Functions.mult(Math.max(0, 1 - tlam.get(4) / norm2(tempVec))));
                etaNorms += norm2(tempVec);
            }
        }
        
//        System.out.println("\ntheta: " + par.theta.toString());
//        System.out.println("\neta: " + par.eta.toString());

        pX.assign(par.toMatrix1D());
        return lambda.get(0)*betaNorms + lambda.get(1)*thetaNorms + lambda.get(2)*phiNorms + lambda.get(3)*gammaNorms + lambda.get(4)*etaNorms;
    }
    
    /*
     * 
     * -survivalmgm -steps 20 -low 0.1 -high 0.6 -d /home/t-love01/Documents/research/censored_variables/simulation_0_10000_data.txt -o /home/t-love01/Documents/research/censored_variables/simulation_0_10000_lasso_out.txt -sif /home/t-love01/Documents/research/censored_variables/simulation_0_10000_lasso_graph.sif
     * 
     */

        /*public Matrix compute(double t, Matrix X){
            double[][] out = new double[1][];
            out[0] = computeColt(t, factory1D.make(X.getContinuousData()[0])).toArray();
            return new DenseMatrix(out);
        }*/

    /*public static class softThreshold implements DoubleFunction{
        public double th;
        public softThreshold(double th){
            this.th = Math.abs(th);
        }

        public double apply(double x){
            if(x > th){
                return x-th;
            } else if(x < -th){
                return x+th;
            } else {
                return 0;
            }
        }
    }*/


    public void setTimeout(long time)
    {
        timeout = time;
    }

    /**
     *  Learn MGM traditional way with objective function tolerance. Recommended for inference applications that need
     *  accurate pseudolikelihood
     *
     * @param epsilon tolerance in change of objective function
     * @param iterLimit iteration limit
     */
    public void learn(double epsilon, int iterLimit){
        ProximalGradient pg = new ProximalGradient();
        setParams(new survivalMGMParams(pg.learnBackTrack(this, params.toMatrix1D(), epsilon, iterLimit), p, lsum, r));
//        while (true) {
//            try {
//                setParams(new survivalMGMParams(pg.learnBackTrack(this, params.toMatrix1D(), epsilon, iterLimit), p, lsum, r));
//                break;
//            } catch (Exception e) {
//                System.out.println(e);
//                e.printStackTrace();
//                System.out.println("retrying");
//            }
//        }
    }

    /**
     *  Learn MGM traditional way with objective function tolerance. Recommended for inference applications that need
     *  accurate pseudolikelihood
     *
     * @param lambdas the lambdas on the solution path
     * @param iterLimit iteration limit
     */
    public ArrayList<survivalMGMParams> learnPath(double epsilon, int iterLimit, double[] lambdas){
        ArrayList<survivalMGMParams> path = new ArrayList<>(lambdas.length);
        for (int i = 0; i < lambdas.length; i++) path.add(new survivalMGMParams());
        ProximalGradient pg = new ProximalGradient(.5, .9, true);
        for (int i = lambdas.length-1; i >= 0; i--) {
            System.out.println("Solving lambda = " + lambdas[i]);
            lambda.assign(lambdas[i]);
            path.set(i, new survivalMGMParams(pg.learnBackTrack(this, params.toMatrix1D(), epsilon, iterLimit), p, lsum, r));
            System.out.println("hot start params: " + path.get(i));
            setParams(path.get(i));
        }
        return path;
    }

    /**
     *  Learn MGM traditional way with objective function tolerance. Recommended for inference applications that need
     *  accurate pseudolikelihood
     *
     * @param lambdas the lambdas on the solution path
     * @param iterLimit iteration limit
     */
    public ArrayList<survivalMGMParams> learnPathEdges(int iterLimit, double[] lambdas) {
        ArrayList<survivalMGMParams> path = new ArrayList<>(lambdas.length);
        for (int i = 0; i < lambdas.length; i++) path.add(new survivalMGMParams());
        ProximalGradient pg = new ProximalGradient(.5, .9, true);
        for (int i = lambdas.length - 1; i >= 0; i--) {
            System.out.println("Solving lambda = " + lambdas[i]);
            lambda.assign(lambdas[i]);
            path.set(i, new survivalMGMParams(pg.learnBackTrack(this, params.toMatrix1D(), 0.0, iterLimit), p, lsum, r));
            setParams(path.get(i));
        }
        return path;
    }

    /**
     *  Learn MGM using edge convergence using default 3 iterations of no edge changes. Recommended when we only care about
     *  edge existence.
     *
     * @param iterLimit
     */
    public void learnEdges(int iterLimit){
        ProximalGradient pg = new ProximalGradient(.5, .9, true);
        if(timeout!=-1)
            setParams(new survivalMGMParams(pg.learnBackTrack(this, params.toMatrix1D(), 0.0, iterLimit,timeout), p, lsum, r));
        else
            setParams(new survivalMGMParams(pg.learnBackTrack(this, params.toMatrix1D(), 0.0, iterLimit), p, lsum, r));
//        while (true) {
//            try {
//                if(timeout!=-1)
//                    setParams(new survivalMGMParams(pg.learnBackTrack(this, params.toMatrix1D(), 0.0, iterLimit,timeout), p, lsum, r));
//                else
//                    setParams(new survivalMGMParams(pg.learnBackTrack(this, params.toMatrix1D(), 0.0, iterLimit), p, lsum, r));
//                break;
//            } catch (Exception e) {
//                System.out.println(e);
//                e.printStackTrace();
//                System.out.println("retrying");
//            }
//        }

        timePerIter = pg.timePerIter;
        iterCount = pg.iterComplete;
//      System.out.println(params.eta);
    }

    /**
     *  Learn MGM using edge convergence using edgeChangeTol (see ProximalGradient for documentation). Recommended when we only care about
     *  edge existence.
     *
     * @param iterLimit
     * @param edgeChangeTol
     */
    public void learnEdges(int iterLimit, int edgeChangeTol){
        ProximalGradient pg = new ProximalGradient(.5, .9, true);
        pg.setEdgeChangeTol(edgeChangeTol);
        setParams(new survivalMGMParams(pg.learnBackTrack(this, params.toMatrix1D(), 0.0, iterLimit), p, lsum, r));
//        while (true) {
//            try {
//                setParams(new survivalMGMParams(pg.learnBackTrack(this, params.toMatrix1D(), 0.0, iterLimit), p, lsum, r));
//                break;
//            } catch (Exception e) {
//                System.out.println(e);
//                e.printStackTrace();
//                System.out.println("retrying");
//            }
//        }
    }

    /**
     * Converts MGM object to Graph object with edges if edge parameters are non-zero. Loses all edge param information
     *
     * @return
     */
    public Graph graphFromMGM(){
        //List<Node> variables = getVariables();
        Graph g = new EdgeListGraph(variables);

        for (int i = 0; i < p; i++) {
            for (int j = i+1; j < p; j++) {
                double v1 = params.beta.get(i, j);

                if (Math.abs(v1)>0) {
                    if (!g.isAdjacentTo(variables.get(i), variables.get(j))) {
                        g.addUndirectedEdge(variables.get(i), variables.get(j));
                    }
                }
            }
        }

        for (int i = 0; i < p; i++) {
            for (int j = 0; j < q; j++) {
                double v1 = params.theta.viewColumn(i).viewPart(lcumsum[j], l[j]).copy().assign(Functions.abs).zSum();

                if (v1>0) {
                    if (!g.isAdjacentTo(variables.get(i), variables.get(p+j))) {
                        g.addUndirectedEdge(variables.get(i), variables.get(p+j));
                    }
                }
            }
        }

        for (int i = 0; i < q; i++) {
            for (int j = i+1; j < q; j++) {
                double v1 = params.phi.viewPart(lcumsum[i], lcumsum[j], l[i], l[j]).copy().assign(Functions.abs).zSum();

                if (v1>0) {
                    if (!g.isAdjacentTo(variables.get(p+i), variables.get(p+j))) {
                        g.addUndirectedEdge(variables.get(p+i), variables.get(p+j));
                    }
                }
            }
        }

        for (int j = 0; j < r; j++) {

            for (int i = 0; i < p; i++) {
                double v1 = params.gamma.get(i, j);

                if (v1 != 0) {
                    if (!g.isAdjacentTo(variables.get(i), variables.get(p + q + j))) {
                        g.addUndirectedEdge(variables.get(i), variables.get(p + q + j));
                    }
                }
            }

            for (int i = 0; i < q; i++) {
                double v1 = params.eta.viewColumn(j).viewPart(lcumsum[i], l[i]).copy().assign(Functions.abs).zSum();

                if (v1 != 0) {
                    if (!g.isAdjacentTo(variables.get(p + i), variables.get(p + q + j))) {
                        g.addUndirectedEdge(variables.get(p + i), variables.get(p + q + j));
                    }
                }
            }
        }

        return g;
    }

    public void moralizeCensoredNeighbors(DataSet ds, Graph g, double alpha) {
        IndependenceTest test = new IndTestMultinomialAJ(ds, alpha, true);
        List<Node> neighbors = new ArrayList<>();
        List<Node> zList = new ArrayList<>();
        List<Node> source = new ArrayList<>();
        List<Node> target = new ArrayList<>();

        for (int j = 0; j < r; j++) {
            neighbors.clear();

            for (Node n : variables) {
                if (g.isAdjacentTo(n, variables.get(p+q+j))) neighbors.add(n);
            }

            if (neighbors.isEmpty()) continue;

//            System.out.println(j + ":\t" + neighbors.size() + " neighbors");

            boolean skip;
            for (int i = 0; i < neighbors.size()-1; i++) {
                for (int k = i+1; k < neighbors.size(); k++) {
                    if (!g.isAdjacentTo(neighbors.get(i), neighbors.get(k))) {
//                        System.out.println(j + ":\t" + neighbors.get(i) + ", " + neighbors.get(k) + " not adjacent");
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
                        zList.add(variables.get(p + q + j));
                        for (Node n : variables) {
                            if (g.isAdjacentTo(n, neighbors.get(i)) || g.isAdjacentTo(n, neighbors.get(k))) {
                                if (!(zList.contains(n) || n instanceof CensoredVariable)) {
                                    zList.add(n);
                                }
                            }
                        }
//                        System.out.println(j + ":\tzList " + zList);
                        if (test.isDependent(neighbors.get(i), neighbors.get(k), zList)) {
//                            System.out.println(j + ":\t" + neighbors.get(i) + ", " + neighbors.get(k) + " are dependent");
                            System.out.println("Edge added between " + neighbors.get(i) + " and " + neighbors.get(k));

                            source.add(neighbors.get(i));
                            target.add(neighbors.get(k));
//                            g.addUndirectedEdge(neighbors.get(i), neighbors.get(k));
                        }
                    }
                }
            }
        }

        for (int i = 0; i < source.size(); i++) g.addUndirectedEdge(source.get(i), target.get(i));
    }

    /**
     * Converts MGM to matrix of doubles. uses 2-norm to combine c-d edge parameters into single value and f-norm for
     * d-d edge parameters.
     *
     * @return
     */
    public DoubleMatrix2D adjMatFromMGM(){
        //List<Node> variables = getVariables();
        DoubleMatrix2D outMat = DoubleFactory2D.dense.make(p+q+r,p+q+r);

        outMat.viewPart(0,0,p,p).assign(params.beta.copy().assign(alg.transpose(params.beta), Functions.plus));

        for (int i = 0; i < p; i++) {
            for (int j = 0; j < q; j++) {
                double val = norm2(params.theta.viewColumn(i).viewPart(lcumsum[j], l[j]));
                outMat.set(i, p + j, val);
                outMat.set(p + j, i, val);
            }
        }

        for (int i = 0; i < q; i++) {
            for (int j = i+1; j < q; j++) {
                double val = alg.normF(params.phi.viewPart(lcumsum[i], lcumsum[j], l[i], l[j]));
                outMat.set(p+i,p+j,val);
                outMat.set(p+j,p+i,val);
            }
        }
        
//        for (int i = 0; i < r; i++) {
//        	for (int j = 0; j < p; j++) {
//        		outMat.set(p + q + i, j, params.gamma.get(j,i));
//        		outMat.set(j, p + q + i, params.gamma.get(j,i));
//        	}
//        }
        
        outMat.viewPart(0, p+q, p, r).assign(params.gamma);
        outMat.viewPart(p+q, 0, r, p).assign(alg.transpose(params.gamma));
        
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < q; j++) {
                double val = norm2(params.eta.viewColumn(i).viewPart(lcumsum[j], l[j]));
                outMat.set(p + q + i, p + j, val);
                outMat.set(p + j, p + q + i, val);
            }
        }

        //order the adjmat to be the same as the original DataSet variable ordering
        if(initVariables!=null) {
            int[] varMap = new int[p+q+r];
            for(int i = 0; i < p+q+r; i++){
                varMap[i] = variables.indexOf(initVariables.get(i));
            }
            outMat = outMat.viewSelection(varMap, varMap);
        }

        return outMat;
    }

    /**
     * Simple search command for GraphSearch implementation. Uses default edge convergence, 1000 iter limit.
     *
     * @return
     */
    public Graph search(){
        long startTime = System.currentTimeMillis();
        learnEdges(500); //unlikely to hit this limit
        elapsedTime = System.currentTimeMillis() - startTime;
        return graphFromMGM();
    }

    /**
     * Return time of execution for learning.
     * @return
     */
    public long getElapsedTime(){
        return elapsedTime;
    }


    /*
     * PRIVATE UTILS
     */
    //Utils
    //sum rows together if marg == 1 and cols together if marg == 2
    //Using row-major speeds up marg=1 5x
    private static DoubleMatrix1D margSum(DoubleMatrix2D mat, int marg){
        int n = 0;
        DoubleMatrix1D vec = null;
        DoubleFactory1D fac = DoubleFactory1D.dense;

        if(marg==1){
            n = mat.columns();
            vec = fac.make(n);
            for (int j = 0; j < mat.rows(); j++){
                for (int i = 0; i < n; i++){
                    vec.setQuick(i, vec.getQuick(i) + mat.getQuick(j,i));
                }
            }
        } else if (marg ==2){
            n = mat.rows();
            vec = fac.make(n);
            for (int i = 0; i < n; i++) {
                vec.setQuick(i, mat.viewRow(i).zSum());
            }
        }

        return vec;
    }

    //zeros out everthing below di-th diagonal
    public static DoubleMatrix2D upperTri(DoubleMatrix2D mat, int di){
        for(int i = Math.max(-di + 1, 0); i < mat.rows(); i++){
            for(int j = 0; j < Math.min(i + di, mat.rows()); j++){
                mat.set(i,j,0);
            }
        }

        return mat;
    }

    //zeros out everthing above di-th diagonal
    private static DoubleMatrix2D lowerTri(DoubleMatrix2D mat, int di){
        for(int i = 0; i < mat.rows() - Math.max(di + 1, 0); i++){
            for(int j = Math.max(i + di + 1, 0); j <  mat.rows(); j++){
                mat.set(i,j,0);
            }
        }

        return mat;
    }

    // should move somewhere else...
    private static double norm2(DoubleMatrix2D mat){
//        return Math.sqrt(mat.copy().assign(Functions.pow(2)).zSum());
        Algebra al = new Algebra();

//        if (Thread.currentThread().getName() == "ForkJoinPool.commonPool-worker-15")
//            System.out.println("####" + Thread.currentThread().getName() + " Determinant: " + al.det(mat));

        //norm found by svd so we need rows >= cols
        if(mat.rows() < mat.columns()){
            return al.norm2(al.transpose(mat));
        }
        return al.norm2(mat);
    }

    private static double norm2(DoubleMatrix1D vec){
//        return Math.sqrt(vec.copy().assign(Functions.pow(2)).zSum());
        return Math.sqrt(new Algebra().norm2(vec));
    }
/*
    private static void runTests1(){
        try {
            //DoubleMatrix2D xIn = DoubleFactory2D.dense.make(loadDataSelect("/Users/ajsedgewick/tetrad/test_data", "med_test_C.txt"));
            //DoubleMatrix2D yIn = DoubleFactory2D.dense.make(loadDataSelect("/Users/ajsedgewick/tetrad/test_data", "med_test_D.txt"));
            //String path = MGM.class.getResource("test_data").getPath();
            String path = "/Users/ajsedgewick/tetrad_master/tetrad/tetrad-lib/src/main/java/edu/pitt/csb/mgm/test_data";
            DoubleMatrix2D xIn = DoubleFactory2D.dense.make(MixedUtils.loadDelim(path, "med_test_C.txt").getDoubleData().toArray());
            DoubleMatrix2D yIn = DoubleFactory2D.dense.make(MixedUtils.loadDelim(path, "med_test_D.txt").getDoubleData().toArray());
            int[] L = new int[24];
            Node[] vars = new Node[48];
            for(int i = 0; i < 24; i++){
                L[i] = 2;
                vars[i] = new ContinuousVariable("X" + i);
                vars[i+24] = new DiscreteVariable("Y" + i);
            }

            double lam = .2;
            MGM model = new MGM(xIn, yIn, new ArrayList<Node>(Arrays.asList(vars)), L, new double[]{lam, lam, lam});
            MGM model2 = new MGM(xIn, yIn, new ArrayList<Node>(Arrays.asList(vars)), L, new double[]{lam, lam, lam});

            System.out.println("Weights: " + Arrays.toString(model.weights.toArray()));

            DoubleMatrix2D test = xIn.copy();
            DoubleMatrix2D test2 = xIn.copy();
            long t = System.currentTimeMillis();
            for(int i=0; i<50000; i++) {
                test2 = xIn.copy();
                test.assign(test2);
            }
            System.out.println("assign Time: " + (System.currentTimeMillis() - t));

            t = System.currentTimeMillis();
            double[][] xArr = xIn.toArray();
            for(int i=0; i<50000; i++) {
                //test = DoubleFactory2D.dense.make(xArr);
                test2 = xIn.copy();
                test = test2;
            }
            System.out.println("equals Time: " + (System.currentTimeMillis() - t));


            System.out.println("Init nll: " + model.smoothValue(model.params.toMatrix1D()));
            System.out.println("Init reg term: " + model.nonSmoothValue(model.params.toMatrix1D()));

            t = System.currentTimeMillis();
            model.learnEdges(700);
            //model.learn(1e-7, 700);
            System.out.println("Orig Time: " + (System.currentTimeMillis()-t));

            System.out.println("nll: " + model.smoothValue(model.params.toMatrix1D()));
            System.out.println("reg term: " + model.nonSmoothValue(model.params.toMatrix1D()));

            System.out.println("params:\n" + model.params);
            System.out.println("adjMat:\n" + model.adjMatFromMGM());


        } catch (IOException ex){
            ex.printStackTrace();
        }
    }

    /**
     * test non penalty use cases
     */
    /*
    private static void runTests2(){
        Graph g = GraphConverter.convert("X1-->X2,X3-->X2,X4-->X5");
        //simple graph pm im gen example

        HashMap<String, Integer> nd = new HashMap<>();
        nd.put("X1", 0);
        nd.put("X2", 0);
        nd.put("X3", 4);
        nd.put("X4", 4);
        nd.put("X5", 4);

        g = MixedUtils.makeMixedGraph(g, nd);

        GeneralizedSemPm pm = MixedUtils.GaussianCategoricalPm(g, "Split(-1.5,-.5,.5,1.5)");
      //  System.out.println(pm);

        GeneralizedSemIm im = MixedUtils.GaussianCategoricalIm(pm);
    //    System.out.println(im);

        int samps = 1000;
        DataSet ds = im.simulateDataAvoidInfinity(samps, false);
        ds = MixedUtils.makeMixedData(ds, nd);
        //System.out.println(ds);

        double lambda = 0;
        MGM model = new MGM(ds, new double[]{lambda, lambda, lambda});

    //    System.out.println("Init nll: " + model.smoothValue(model.params.toMatrix1D()));
//        System.out.println("Init reg term: " + model.nonSmoothValue(model.params.toMatrix1D()));

        model.learn(1e-8,1000);

        System.out.println("Learned nll: " + model.smoothValue(model.params.toMatrix1D()));
        System.out.println("Learned reg term: " + model.nonSmoothValue(model.params.toMatrix1D()));

        System.out.println("params:\n" + model.params);
        System.out.println("adjMat:\n" + model.adjMatFromMGM());
    }

    public static void main(String[] args){
        runTests1();
    }
*/
}

