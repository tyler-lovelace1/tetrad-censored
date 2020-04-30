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

import cern.colt.function.DoubleFunction;
import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.jet.math.Functions;
import edu.cmu.tetrad.data.*;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.regression.*;
import edu.cmu.tetrad.search.IndependenceTest;
import edu.cmu.tetrad.search.SearchLogUtils;
import edu.cmu.tetrad.util.Function;
import edu.cmu.tetrad.util.TetradLogger;
import edu.cmu.tetrad.util.TetradMatrix;
import org.apache.commons.math3.distribution.ChiSquaredDistribution;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.*;

/**
 * Performs a test of conditional independence X _||_ Y | Z1...Zn where all searchVariables are either continuous or discrete.
 * This test is valid for both ordinal and non-ordinal discrete searchVariables.
 * <p>
 * This logisticRegression makes multiple assumptions: 1. IIA 2. Large sample size (multiple regressions needed on subsets of
 * sample)
 *
 * @author Joseph Ramsey
 * @author Augustus Mayo.
 */
public class IndTestMultinomialQL implements IndependenceTest {
    public int timesCalled;
    private DataSet originalData;
    private List<Node> searchVariables;
    private DataSet internalData;
    private double alpha;
    private double lastP;
    private Map<Node, List<Node>> variablesPerNode = new HashMap<Node, List<Node>>();
    private LogisticRegression logisticRegression;
    private RegressionDataset regression;
    private CoxRegression coxRegression;
    private boolean verbose = false;
    private DoubleFactory2D factory2D = DoubleFactory2D.dense;
    private DoubleFactory1D factory1D = DoubleFactory1D.dense;
    private boolean preferLinear;

    public int reset()
    {
        int temp = timesCalled;
        timesCalled = 0;
        return temp;
    }
    public IndTestMultinomialQL(DataSet data, double alpha) {
        this.searchVariables = data.getVariables();
        this.originalData = data.copy();
        DataSet internalData = data.copy();
        this.alpha = alpha;

        List<Node> variables = internalData.getVariables();

        for (Node node : variables) {
            List<Node> nodes = expandVariable(internalData, node);
            variablesPerNode.put(node, nodes);
        }

        this.internalData = internalData;
        this.logisticRegression = new LogisticRegression(internalData);
        this.regression = new RegressionDataset(internalData);
        this.coxRegression = new CoxRegression(internalData);

    }
    public IndTestMultinomialQL(DataSet data, double alpha, boolean preferLinear) {
        this.preferLinear = preferLinear;
        this.searchVariables = data.getVariables();
        this.originalData = data.copy();
        DataSet internalData = data.copy();
        this.alpha = alpha;

        List<Node> variables = internalData.getVariables();

        for (Node node : variables) {
            List<Node> nodes = expandVariable(internalData, node);
            variablesPerNode.put(node, nodes);
        }

        this.internalData = internalData;
        this.logisticRegression = new LogisticRegression(internalData);
        this.regression = new RegressionDataset(internalData);
        this.coxRegression = new CoxRegression(internalData);

        int[] _rows = getNonMissingRows(originalData.getVariable(0), originalData.getVariable(1), originalData.getVariables());

        List<Node> cens = new ArrayList<>();
        List<Node> complete = new ArrayList<>();
        for (Node n : originalData.getVariables()) {
            if (n instanceof CensoredVariable) cens.add(n);
            else complete.add(n);
        }
        System.out.println(cens);
        System.out.println(complete);
        estimateExpVal(cens, complete, _rows);
    }

    /**
     * @return an Independence test for a subset of the searchVariables.
     */
    public IndependenceTest indTestSubset(List<Node> vars) {
        throw new UnsupportedOperationException();
    }

    /**
     * @return true if the given independence question is judged true, false if not. The independence question is of the
     * form x _||_ y | z, z = <z1,...,zn>, where x, y, z1,...,zn are searchVariables in the list returned by
     * getVariableNames().
     */
    public boolean isIndependent(Node x, Node y, List<Node> z) {
        timesCalled++;
        //TODO find out if this is a big problem
        if(z==null)
            z = Collections.emptyList();
        if (x instanceof CensoredVariable) {
            if (y instanceof CensoredVariable) {
                return true; // Assumed non-interacting at first
            }
            return isIndependentCoxRegression(x, y, z);
        } else if (y instanceof CensoredVariable) {
            return isIndependentCoxRegression(y, x, z);
        } else if (x instanceof DiscreteVariable) {
            return isIndependentMultinomialLogisticRegression(x, y, z);
        } else if (y instanceof DiscreteVariable) {
            if(preferLinear)
            {
                return isIndependentRegression(x,y,z);
            }
            else {
                return isIndependentMultinomialLogisticRegression(y, x, z);
            }
        } else {
            return isIndependentRegression(x, y, z);
        }
    }

    private synchronized List<Node> expandVariable(DataSet dataSet, Node node) {
        if (node instanceof ContinuousVariable) {
            return Collections.singletonList(node);
        }

        if (node instanceof CensoredVariable) {
            return Collections.singletonList(node);
        }

        if (node instanceof DiscreteVariable && ((DiscreteVariable) node).getNumCategories() < 3) {
            return Collections.singletonList(node);
        }

        if (!(node instanceof DiscreteVariable)) {
            throw new IllegalArgumentException();
        }

        List<String> varCats = new ArrayList<String>(((DiscreteVariable) node).getCategories());

        // first category is reference
        varCats.remove(0);
        List<Node> variables = new ArrayList<Node>();

        for (String cat : varCats) {

            Node newVar;

            do {
                String newVarName = node.getName() + "MULTINOM" + "." + cat;
                newVar = new DiscreteVariable(newVarName, 2);
            } while (dataSet.getVariable(newVar.getName()) != null);

            variables.add(newVar);

            dataSet.addVariable(newVar);
            int newVarIndex = dataSet.getColumn(newVar);
            int numCases = dataSet.getNumRows();

            for (int l = 0; l < numCases; l++) {
                Object dataCell = dataSet.getObject(l, dataSet.getColumn(node));
                int dataCellIndex = ((DiscreteVariable) node).getIndex(dataCell.toString());

                if (dataCellIndex == ((DiscreteVariable) node).getIndex(cat))
                    dataSet.setInt(l, newVarIndex, 1);
                else
                    dataSet.setInt(l, newVarIndex, 0);
            }
        }

        return variables;
    }

    private boolean isIndependentMultinomialLogisticRegression(Node x, Node y, List<Node> z) {
        if (!variablesPerNode.containsKey(x)) {
            throw new IllegalArgumentException("Unrecogized node: " + x);
        }

        if (!variablesPerNode.containsKey(y)) {
            throw new IllegalArgumentException("Unrecogized node: " + y);
        }

        for (Node node : z) {
            if (!variablesPerNode.containsKey(x)) {
                throw new IllegalArgumentException("Unrecogized node: " + node);
            }
        }

        List<Double> pValues = new ArrayList<Double>();

        List<Node> yzList = new ArrayList<>();
        List<Node> zList = new ArrayList<>();

//        DoubleMatrix2D expVals = factory2D.make(1, 1, 0.0);

        yzList.addAll(variablesPerNode.get(y));
        boolean censregressors = y instanceof CensoredVariable;
        for (Node _z : z) {
            if (_z instanceof CensoredVariable) censregressors=true;
            yzList.addAll(variablesPerNode.get(_z));
            zList.addAll(variablesPerNode.get(_z));
        }

        int[] _rows = getNonMissingRows(x, y, z);

        List<Node> cens = new ArrayList<>();
        if (censregressors) {
//            List<Node> complete = new ArrayList<>();
            for (Node n : yzList) {
                if (n instanceof CensoredVariable) cens.add(n);
//                else complete.add(n);
            }
//            expVals = estimateExpVal(cens, complete, _rows);
        }

//        logisticRegression.setRows(_rows);

        DoubleMatrix2D expVals = factory2D.make(_rows.length, cens.size(),0);
        if (censregressors) {
            for (int i = 0; i < cens.size(); i++) {
                expVals.viewColumn(i).assign(((CensoredVariable) cens.get(i)).getExpval());
            }
        }

        //double[][] coeffsDep = new double[variablesPerNode.get(x).size()][];
        DoubleMatrix2D coeffsNull = DoubleFactory2D.dense.make(zList.size()+1, variablesPerNode.get(x).size());
        DoubleMatrix2D coeffsDep = DoubleFactory2D.dense.make(yzList.size()+1, variablesPerNode.get(x).size());


        for (int i = 0; i < variablesPerNode.get(x).size(); i++) {
            Node _x = variablesPerNode.get(x).get(i);
            // Without y
            //List<Node> regressors0 = new ArrayList<Node>();

            //for (Node _z : z) {
            //regressors0.addAll(variablesPerNode.get(_z));
            //}



            // With y.
            /*List<Node> regressors1 = new ArrayList<Node>();
            regressors1.addAll(variablesPerNode.get(y));
            for (Node _z : z) {
                regressors1.addAll(variablesPerNode.get(_z));
            }*/

            LogisticRegression.Result result0;
            LogisticRegression.Result result1;

            if (!censregressors) {
                result0 = logisticRegression.regress((DiscreteVariable) _x, zList, _rows);
                result1 = logisticRegression.regress((DiscreteVariable) _x, yzList, _rows);
            } else {
                result0 = logisticRegression.regress((DiscreteVariable) _x, zList, _rows, cens, expVals.viewDice().toArray());
                result1 = logisticRegression.regress((DiscreteVariable) _x, yzList, _rows, cens, expVals.viewDice().toArray());
            }

            coeffsNull.viewColumn(i).assign(result0.getCoefs());
            coeffsDep.viewColumn(i).assign(result1.getCoefs());

            // Returns -2 LL
            //double ll0 = result0.getLogLikelihood();
            //double ll1 = result1.getLogLikelihood();

            //double chisq = (ll0 - ll1);
            //int df = variablesPerNode.get(y).size();
            //double p = 1.0 - new ChiSquaredDistribution(df).cumulativeProbability(chisq);
            //pValues.add(p);
        }

        double chisq = 2*(multiLL(coeffsDep, x, yzList, _rows) - multiLL(coeffsNull, x, zList, _rows));
        int df = variablesPerNode.get(y).size()*variablesPerNode.get(x).size();
//        System.out.println("chisq: " + chisq);
        if (Double.isNaN(chisq) || chisq < 0) chisq = 0.0;
        double p = 1.0 - new ChiSquaredDistribution(df).cumulativeProbability(chisq);

        //double p = 1.0;

        // Choose the minimum of the p-values
        // This is only one method that can be used, this requires every coefficient to be significant
        //for (double val : pValues) {
        //    if (val < p) p = val;
        //}

        boolean indep = p > alpha;

        this.lastP = p;

        if (verbose) {
            if (indep) {
                TetradLogger.getInstance().log("independencies", SearchLogUtils.independenceFactMsg(x, y, z, p));
            } else {
                TetradLogger.getInstance().log("dependencies", SearchLogUtils.dependenceFactMsg(x, y, z, p));
            }
        }

        //t.println(x + " is independent of " + y + " given " + z + ": " + indep);
        return indep;
    }

    int[] _rows = null;

    // This takes an inordinate amount of time. -jdramsey 20150929
    private int[] getNonMissingRows(Node x, Node y, List<Node> z) {
        List<Integer> rows = new ArrayList<Integer>();
        boolean missing;

        for (int i = 0; i < internalData.getNumRows(); i++) {
            missing = false;
            for (Node node : variablesPerNode.get(x)) {
                if (isMissing(node, i)) {
                    missing=true;
                    break;
                }
            }

            if (missing) continue;

            for (Node node : variablesPerNode.get(y)) {
                if (isMissing(node, i)) {
                    missing = true;
                    break;
                }
            }

            if (missing) continue;

            for (Node _z : z) {
                for (Node node : variablesPerNode.get(_z)) {
                    if (isMissing(node, i)) {
                        missing = true;
                        break;
                    }
                }
            }

            if (missing) continue;

            rows.add(i);
        }

//        System.out.println("getnonmissingrows: CC = " + rows.size());
        int[] _rows = new int[rows.size()];
        for (int k = 0; k < rows.size(); k++) _rows[k] = rows.get(k);

//        if (_rows == null) {
//            _rows = new int[internalData.getNumRows()];
//            for (int k = 0; k < _rows.length; k++) _rows[k] = k;
//        }

        return _rows;
    }

    private boolean isMissing(Node x, int i) {
        int j = internalData.getColumn(x);

        if (x instanceof DiscreteVariable) {
            int v = internalData.getInt(i, j);

            if (v == -99) {
                return true;
            }
        }

        if (x instanceof ContinuousVariable) {
            double v = internalData.getDouble(i, j);

            if (Double.isNaN(v)) {
                return true;
            }
        }

        return false;
    }

    private double multiLL(DoubleMatrix2D coeffs, Node dep, List<Node> indep, int[] _rows){

        if(dep == null) throw new IllegalArgumentException("must have a dependent node to regress on!");
        List<Node> depList = new ArrayList<>();
        depList.add(dep);
        DoubleMatrix2D depData = factory2D.make(internalData.subsetColumns(depList).subsetRows(_rows).getDoubleData().toArray());
        int N = depData.rows();

        DoubleMatrix2D indepData;
        if(indep.size()==0)
            indepData = factory2D.make(N,1,1.0);
        else {
            indepData = factory2D.make(internalData.subsetColumns(indep).subsetRows(_rows).getDoubleData().toArray());
            indepData = factory2D.appendColumns(factory2D.make(N, 1, 1.0), indepData);
        }

        DoubleMatrix2D probs = Algebra.DEFAULT.mult(indepData, coeffs);

        probs = factory2D.appendColumns(factory2D.make(indepData.rows(), 1, 1.0), probs).assign(Functions.exp);
        double ll = 0;
        for(int i = 0; i < N; i++){
            DoubleMatrix1D curRow = probs.viewRow(i);
            curRow.assign(Functions.div(curRow.zSum()));
            ll += Math.log(curRow.get((int)depData.get(i,0)));
        }
        return ll;
    }

    private boolean isIndependentRegression(Node x, Node y, List<Node> z) {
        if (!variablesPerNode.containsKey(x)) {
            throw new IllegalArgumentException("Unrecogized node: " + x);
        }

        if (!variablesPerNode.containsKey(y)) {
            throw new IllegalArgumentException("Unrecogized node: " + y);
        }

        for (Node node : z) {
            if (!variablesPerNode.containsKey(x)) {
                throw new IllegalArgumentException("Unrecogized node: " + node);
            }
        }

//        DoubleMatrix2D expVals = factory2D.make(1,1,0);
        List<Node> regressors = new ArrayList<Node>();
        regressors.add(internalData.getVariable(y.getName()));

        boolean censregressors = y instanceof CensoredVariable;
        for (Node _z : z) {
            if (_z instanceof CensoredVariable) censregressors=true;
            regressors.addAll(variablesPerNode.get(_z));
        }

        int[] _rows = getNonMissingRows(x, y, z);
        regression.setRows(_rows);

        List<Node> cens = new ArrayList<>();
        if (censregressors) {
//            List<Node> complete = new ArrayList<>();
            for (Node n : regressors) {
                if (n instanceof CensoredVariable) cens.add(n);
//                else complete.add(n);
            }
//            expVals = estimateExpVal(cens, complete, _rows);
        }

        DoubleMatrix2D expVals = factory2D.make(_rows.length, cens.size(),0);
        if (censregressors) {
            for (int i = 0; i < cens.size(); i++) {
                expVals.viewColumn(i).assign(((CensoredVariable) cens.get(i)).getExpval());
            }
        }

        RegressionResult result;

        try {
            if (!censregressors) result = regression.regress(x, regressors);
            else result = regression.regress(x, regressors, cens, expVals);
        } catch (Exception e) {
            return false;
        }

        double p = result.getP()[1];
        this.lastP = p;

        boolean indep = p > alpha;

        if (verbose) {
            if (indep) {
                TetradLogger.getInstance().log("independencies", SearchLogUtils.independenceFactMsg(x, y, z, p));
            } else {
                TetradLogger.getInstance().log("dependencies", SearchLogUtils.dependenceFactMsg(x, y, z, p));
            }
        }

        //System.out.println(x + " is independent of " + y + " given " + z + ":" + indep);
        return indep;
    }

    private boolean isIndependentCoxRegression(Node x, Node y, List<Node> z) {
        if (!variablesPerNode.containsKey(x)) {
            throw new IllegalArgumentException("Unrecogized node: " + x);
        }

        if (!variablesPerNode.containsKey(y)) {
            throw new IllegalArgumentException("Unrecogized node: " + y);
        }

        for (Node node : z) {
            if (!variablesPerNode.containsKey(x)) {
                throw new IllegalArgumentException("Unrecogized node: " + node);
            }
        }

//        DoubleMatrix2D expVals = factory2D.make(1,1,0);
        List<Node> zList = new ArrayList<>();
        List<Node> yzList = new ArrayList<>();
        if (y instanceof DiscreteVariable) {
            yzList.addAll(variablesPerNode.get(y));
        } else {
            yzList.add(internalData.getVariable(y.getName()));
        }

        System.out.println("y: " + yzList);

        boolean censregressors = y instanceof CensoredVariable;
        for (Node _z : z) {
            if (_z instanceof CensoredVariable) censregressors=true;
            yzList.addAll(variablesPerNode.get(_z));
            if (y instanceof DiscreteVariable) zList.addAll(variablesPerNode.get(_z));
        }

        int[] _rows = getNonMissingRows(x, y, z);

        List<Node> cens = new ArrayList<>();
        if (censregressors) {
//            List<Node> complete = new ArrayList<>();
            for (Node n : yzList) {
                if (n instanceof CensoredVariable) cens.add(n);
//                else complete.add(n);
            }
//            expVals = estimateExpVal(cens, complete, _rows);
        }

        DoubleMatrix2D expVals = factory2D.make(_rows.length, cens.size(),0);
        if (censregressors) {
            for (int i = 0; i < cens.size(); i++) {
                expVals.viewColumn(i).assign(((CensoredVariable) cens.get(i)).getExpval());
            }
        }

//        int[] _rows = getNonMissingRows(x, y, z);
        System.out.println("complete samples: " + _rows.length);
//        coxRegression.setRows(_rows);

        CoxRegressionResult result;
//        CoxRegressionResult result0;

        if (!censregressors) result = coxRegression.regress((CensoredVariable) x, yzList, _rows);
        else result = coxRegression.regress((CensoredVariable) x, yzList, _rows, cens, expVals);

        double p = 1.0;
        if (y instanceof DiscreteVariable) {

            CoxRegressionResult result0;
            if (!censregressors) result0 = coxRegression.regress((CensoredVariable) x, zList, _rows);
            else result0 = coxRegression.regress((CensoredVariable) x, zList, _rows, cens, expVals);

            double ll = result.getLoglikelihood();
            double ll0 = result0.getLoglikelihood();
            System.out.println("likelihood ratio: " + (ll - ll0));
            p -= new ChiSquaredDistribution(variablesPerNode.get(y).size()).cumulativeProbability(2 * (ll - ll0));
        }
        else p = result.getP()[0];

        this.lastP = p;

        boolean indep = p > alpha;

//        System.out.println("likelihood ratio: " + 2 * (ll - ll0));
        System.out.println("p-value: " + p);
        System.out.println(x.getName() + " and " + y.getName() + " are independent given " + z + ": " + indep);

        if (verbose) {
            if (indep) {
                TetradLogger.getInstance().log("independencies", SearchLogUtils.independenceFactMsg(x, y, z, p));
            } else {
                TetradLogger.getInstance().log("dependencies", SearchLogUtils.dependenceFactMsg(x, y, z, p));
            }
        }

        //System.out.println(x + " is independent of " + y + " given " + z + ":" + indep);
        return indep;
    }


    private void estimateExpVal(List<Node> cens, List<Node> complete, int[] _rows) {
        Algebra alg = new Algebra();
        DoubleMatrix2D expVals = factory2D.make(_rows.length, cens.size(), 0.0);
        DoubleMatrix1D X;
        DoubleMatrix2D Z;
        DoubleMatrix1D beta;
        double U;
        double T;
        double P;
        System.out.println("estimating expected value");
        int col = 0;
        for (Node _n : cens) {
            CoxRegressionResult result = coxRegression.regress((CensoredVariable) _n, complete, _rows);
//            System.out.println("estimated beta:");
//            for (int i = 0; i < result.getCoef().length; i++) System.out.print(" " + result.getCoef()[i]);
//            System.out.println();
            double[] weib_params = estimateWeibull((CensoredVariable) _n, complete, result.getCoef());
            double[] expectedTime = new double[((CensoredVariable) _n).getCensor().length];
            List<Node> censList = new ArrayList<>();
            censList.add(_n);
            X = factory2D.make(internalData.subsetColumns(censList).getDoubleData().toArray()).viewColumn(0);
            if (complete.size() == 0) {
                for (int i = 0; i < _rows.length; i++) {
                    if (((CensoredVariable) _n).getCensor(_rows[i]) == 0) {
//                        System.out.println("reported time: " + X.get(i));
                        U = Math.exp(-Math.pow(X.get(_rows[i]) / weib_params[0], weib_params[1]));
//                        System.out.println("reported percentile: " + U);
//                        P = U + (1-U)/2.0;
                        P = U / 2.0;
//                        System.out.println("expected percentile: " + P);
                        T =  weib_params[0] * Math.pow(-Math.log(P), 1/weib_params[1]);
//                        System.out.println("expected time: " + T);
//                        expVals.set(i, col, T);
                        expectedTime[i] = T;
                        if (T < X.get(i)) {
                            System.out.println("Emergency nonsense");
                            System.out.println("Weibull params: " + weib_params[0] + "\t" + weib_params[1]);
                            System.out.println("reported time: " + X.get(i));
                            System.out.println("reported percentile: " + U);
                            System.out.println("expected percentile: " + P);
                            System.out.println("expected time: " + T);
                        }
                    }
                    else {
//                        expVals.set(i, col, X.get(_rows[i]));
                        expectedTime[i] = X.get(_rows[i]);
                    }
                }
            } else {
                beta = factory1D.make(result.getCoef());
                Z = factory2D.make(internalData.subsetColumns(complete).getDoubleData().toArray());
                DoubleMatrix1D theta = alg.mult(Z, beta);
                theta.assign(Functions.exp);
                for (int i = 0; i < _rows.length; i++) {
                    if (((CensoredVariable) _n).getCensor(_rows[i]) == 0) {
//                        System.out.println("reported time: " + X.get(i));
                        U = Math.exp(-Math.pow(X.get(_rows[i]) / weib_params[0], weib_params[1]) * theta.get(_rows[i]));
//                        System.out.println("reported percentile: " + (1-U));
//                        P = U + (1-U)/2.0;
                        P = U/2.0;
//                        System.out.println("expected percentile: " + (1-P));
                        T =  weib_params[0] * Math.pow(-Math.log(P) / theta.get(_rows[i]), 1/weib_params[1]);
//                        System.out.println("expected time: " + T);
//                        expVals.set(i, col, T);
                        if (T == Double.POSITIVE_INFINITY) T = 1e8;
                        expectedTime[i] = T;
                        System.out.println(X.get(_rows[i]) + "* --> " + T);
                        if (T < X.get(_rows[i])) {
                            System.out.println("Emergency nonsense");
                            System.out.println("Weibull params: " + weib_params[0] + "\t" + weib_params[1]);
                            System.out.println("reported time: " + X.get(i));
                            System.out.println("reported percentile: " + U);
                            System.out.println("theta[i]: " + theta.get(i));
                            System.out.println("expected percentile: " + P);
                            System.out.println("expected time: " + T);
                        }
                    }
                    else {
//                        expVals.set(i, col, X.get(_rows[i]));
                        expectedTime[i] = X.get(_rows[i]);
                        System.out.println(X.get(_rows[i]));
                    }
                }
            }
            ((CensoredVariable) _n).setExpval(expectedTime);
//            col++;
        }
//        return expVals;
    }

    private double[] estimateWeibull(CensoredVariable _n, List<Node> complete, double[] b) {
        System.out.println("estimating Weibull parameters");
        Algebra alg = new Algebra();
        double[] weib_params = {0.0, 0.0};
        double nu = 0.0;
        double curnu = 10.0;
        double logmean = 0.0;
        double num = 0.0;
        double denom = 0.0;
        int n;
        DoubleMatrix1D X;
        DoubleMatrix2D Z;
        DoubleMatrix1D beta = factory1D.make(b);
        List<Node> varList = new ArrayList<>(complete);
        List<Node> censList = new ArrayList<>();
        varList.add(_n);
        censList.add(_n);
        DataSet tempData = internalData.subsetColumns(varList).subsetRows(_n.getCC());
        X = factory2D.make(tempData.subsetColumns(censList).getDoubleData().toArray()).viewColumn(0);
//        double[][] tempX = tempData.subsetColumns(censList).getDoubleData().toArray();
//        double[] X1D = new double[tempX.length];
//        for (int i =0; i < X1D.length; i++) X1D[i] = tempX[i][0];
//        X = factory1D.make();
        n = X.size();
        if (complete.size() == 0) {
            Z = factory2D.make(tempData.subsetColumns(complete).getDoubleData().toArray());
            DoubleMatrix1D theta = alg.mult(Z, beta);
            theta.assign(Functions.exp);
            while (Math.abs(nu-curnu) > 1e-4) {
                nu = curnu;
                logmean = X.copy().assign(theta.copy().assign(Functions.pow(1/nu)), Functions.mult).assign(Functions.log).zSum() / n;
                num = X.copy().assign(Functions.pow(nu)).assign(theta, Functions.mult).assign(X.copy().assign(theta.copy().assign(Functions.pow(1/nu)), Functions.mult).assign(Functions.log), Functions.mult).zSum();
                denom = X.copy().assign(Functions.pow(nu)).assign(theta, Functions.mult).zSum();
                curnu = 1 / (num/denom - logmean);
            }
            weib_params[1] = curnu;
            weib_params[0] = Math.pow(denom / n, 1/curnu);
        } else{
            logmean = X.copy().assign(Functions.log).zSum() / n;
            while (Math.abs(nu-curnu) > 1e-4) {
                nu = curnu;
                num = X.copy().assign(Functions.pow(nu)).assign(X.copy().assign(Functions.log), Functions.mult).zSum();
                denom = X.copy().assign(Functions.pow(nu)).zSum();
                curnu = 1 / (num/denom - logmean);
            }
            weib_params[1] = curnu;
            weib_params[0] = Math.pow(denom / n, 1/curnu);
        }
        System.out.println("Weibull params: " + weib_params[0] + "\t" + weib_params[1]);
//        System.exit(0);
        return weib_params;
    }


    public boolean isIndependent(Node x, Node y, Node... z) {
        List<Node> zList = Arrays.asList(z);
        return isIndependent(x, y, zList);
    }

    /**
     * @return true if the given independence question is judged false, true if not. The independence question is of the
     * form x _||_ y | z, z = <z1,...,zn>, where x, y, z1,...,zn are searchVariables in the list returned by
     * getVariableNames().
     */
    public boolean isDependent(Node x, Node y, List<Node> z) {
        return !this.isIndependent(x, y, z);
    }

    public boolean isDependent(Node x, Node y, Node... z) {
        List<Node> zList = Arrays.asList(z);
        return isDependent(x, y, zList);
    }

    /**
     * @return the probability associated with the most recently executed independence test, of Double.NaN if p value is
     * not meaningful for tis test.
     */
    public double getPValue() {
        return this.lastP; //STUB
    }

    /**
     * @return the list of searchVariables over which this independence checker is capable of determinining independence
     * relations.
     */
    public List<Node> getVariables() {
        return searchVariables; // Make sure the variables from the ORIGINAL data set are returned, not the modified dataset!
    }

    /**
     * @return the list of variable varNames.
     */
    public List<String> getVariableNames() {
        List<Node> variables = getVariables();
        List<String> variableNames = new ArrayList<String>();
        for (Node variable1 : variables) {
            variableNames.add(variable1.getName());
        }
        return variableNames;
    }

    public Node getVariable(String name) {
        for (int i = 0; i < getVariables().size(); i++) {
            Node variable = getVariables().get(i);
            if (variable.getName().equals(name)) {
                return variable;
            }
        }

        return null;
    }

    /**
     * @return true if y is determined the variable in z.
     */
    public boolean determines(List<Node> z, Node y) {
        return false; //stub
    }

    /**
     * @return the significance level of the independence test.
     * @throws UnsupportedOperationException if there is no significance level.
     */
    public double getAlpha() {
        return this.alpha; //STUB
    }

    /**
     * Sets the significance level.
     */
    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    public DataSet getData() {
        return this.originalData;
    }

    @Override
    public ICovarianceMatrix getCov() {
        return null;
    }

    @Override
    public List<DataSet> getDataSets() {
        return null;
    }

    @Override
    public int getSampleSize() {
        return 0;
    }

    @Override
    public List<TetradMatrix> getCovMatrices() {
        return null;
    }


    public double getScore() {
        return getPValue();
    }

    /**
     * @return a string representation of this test.
     */
    public String toString() {
        NumberFormat nf = new DecimalFormat("0.0000");
        return "Multinomial Logistic Regression, alpha = " + nf.format(getAlpha());
    }

    public boolean isVerbose() {
        return verbose;
    }

    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }
}
