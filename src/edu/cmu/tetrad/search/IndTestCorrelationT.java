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

import edu.cmu.tetrad.data.*;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.util.*;
import org.apache.commons.math3.distribution.TDistribution;

import java.io.PrintStream;
import java.text.NumberFormat;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

import static java.lang.Math.abs;

/**
 * Checks conditional independence of variable in a continuous data set using Fisher's Z test. See Spirtes, Glymour, and
 * Scheines, "Causation, Prediction and Search," 2nd edition, page 94.
 *
 * @author Joseph Ramsey
 * @author Frank Wimberly adapted IndTestCramerT for Fisher's Z
 */
public final class IndTestCorrelationT implements IndependenceTest {

    /**
     * The covariance matrix.
     */
    private final ICovarianceMatrix covMatrix;

    /**
     * The matrix out of the cov matrix.
     */
    private final TetradMatrix _covMatrix;

    /**
     * The variables of the covariance matrix, in order. (Unmodifiable list.)
     */
    private List<Node> variables;

    /**
     * The significance level of the independence tests.
     */
    private double alpha;

    /**
     * The value of the Fisher's Z statistic associated with the las calculated partial correlation.
     */
    private double pValue;

    /**
     * Formats as 0.0000.
     */
    private static NumberFormat nf = NumberFormatUtil.getInstance().getNumberFormat();

    /**
     * Stores a reference to the dataset being analyzed.
     */
    private DataSet dataSet;

    private PrintStream pValueLogger;
    private Map<Node, Integer> indexMap;
    private Map<String, Node> nameMap;
    private TDistribution tDistribution;
    private boolean verbose = false;

    //==========================CONSTRUCTORS=============================//

    /**
     * Constructs a new Independence test which checks independence facts based on the correlation matrix implied by the
     * given data set (must be continuous). The given significance level is used.
     *
     * @param dataSet A data set containing only continuous columns.
     * @param alpha   The alpha level of the test.
     */
    public IndTestCorrelationT(DataSet dataSet, double alpha) {
        if (!(dataSet.isContinuous())) {
            throw new IllegalArgumentException("Data set must be continuous.");
        }

        this.covMatrix = new CovarianceMatrix(dataSet);
        this._covMatrix = covMatrix.getMatrix();
        List<Node> nodes = covMatrix.getVariables();

        this.variables = Collections.unmodifiableList(nodes);
        this.indexMap = indexMap(variables);
        this.nameMap = mapNames(variables);
        setAlpha(alpha);

        this.dataSet = DataUtils.center(dataSet);

        tDistribution = new TDistribution(dataSet.getNumRows() - 2);
    }

    /**
     * Constructs a new Fisher Z independence test with the listed arguments.
     *
     * @param data      A 2D continuous data set with no missing values.
     * @param variables A list of variables, a subset of the variables of <code>data</code>.
     * @param alpha     The significance cutoff level. p values less than alpha will be reported as dependent.
     */
    public IndTestCorrelationT(TetradMatrix data, List<Node> variables, double alpha) {
        this.dataSet = ColtDataSet.makeContinuousData(variables, data);
        this.dataSet = DataUtils.center(dataSet);
        this.covMatrix = new CovarianceMatrix(dataSet);
        this._covMatrix = covMatrix.getMatrix();
        this.variables = Collections.unmodifiableList(variables);
        this.indexMap = indexMap(variables);
        this.nameMap = mapNames(variables);
        setAlpha(alpha);
    }

    /**
     * Constructs a new independence test that will determine conditional independence facts using the given correlation
     * matrix and the given significance level.
     */
    public IndTestCorrelationT(ICovarianceMatrix corrMatrix, double alpha) {
        this.covMatrix = corrMatrix;
        this._covMatrix = corrMatrix.getMatrix();
        this.variables = Collections.unmodifiableList(corrMatrix.getVariables());
        this.indexMap = indexMap(variables);
        this.nameMap = mapNames(variables);
        setAlpha(alpha);
    }

    //==========================PUBLIC METHODS=============================//

    /**
     * Creates a new independence test instance for a subset of the variables.
     */
    public IndependenceTest indTestSubset(List<Node> vars) {
        if (vars.isEmpty()) {
            throw new IllegalArgumentException("Subset may not be empty.");
        }

        for (Node var : vars) {
            if (!variables.contains(var)) {
                throw new IllegalArgumentException(
                        "All vars must be original vars");
            }
        }

        int[] indices = new int[vars.size()];

        for (int i = 0; i < indices.length; i++) {
            indices[i] = indexMap.get(vars.get(i));
        }

        ICovarianceMatrix newCovMatrix = covMatrix.getSubmatrix(indices);

        double alphaNew = getAlpha();
        return new IndTestCorrelationT(newCovMatrix, alphaNew);
    }

    /**
     * Determines whether variable x is independent of variable y given a list of conditioning variables z.
     *
     * @param x the one variable being compared.
     * @param y the second variable being compared.
     * @param z the list of conditioning variables.
     * @return true iff x _||_ y | z.
     * @throws RuntimeException if a matrix singularity is encountered.
     */
    public boolean isIndependent(Node x, Node y, List<Node> z) {
        double r;
        int n = sampleSize();

        TetradMatrix submatrix = DataUtils.subMatrix(_covMatrix, indexMap, x, y, z);
        r = StatUtils.partialCorrelation(submatrix);

        double t = Math.sqrt(n - 2) * (r / Math.sqrt(1. - r * r));
        pValue = 2.0 * (1.0 - gettDistribution().cumulativeProbability(abs(t)));

        boolean independent = pValue > alpha;

        if (verbose) {
            if (independent) {
                TetradLogger.getInstance().log("independencies",
                        SearchLogUtils.independenceFactMsg(x, y, z, r)); //getScore()));
            } else {
                if (pValueLogger != null) {
                    pValueLogger.println(getPValue());
                }

                TetradLogger.getInstance().log("dependencies",
                        SearchLogUtils.dependenceFactMsg(x, y, z, getPValue()));
            }
        }

        return independent;
    }

    public boolean isIndependent(Node x, Node y, Node... z) {
        return isIndependent(x, y, Arrays.asList(z));
    }

    public boolean isDependent(Node x, Node y, List<Node> z) {
        return !isIndependent(x, y, z);
    }

    public boolean isDependent(Node x, Node y, Node... z) {
        List<Node> zList = Arrays.asList(z);
        return isDependent(x, y, zList);
    }

    /**
     * @return the probability associated with the most recently computed independence test.
     */
    public double getPValue() {
        return pValue;
    }

    /**
     * Sets the significance level at which independence judgments should be made.  Affects the cutoff for partial
     * correlations to be considered statistically equal to zero.
     */
    public void setAlpha(double alpha) {
        if (alpha < 0.0 || alpha > 1.0) {
            throw new IllegalArgumentException("Significance out of range.");
        }

        this.alpha = alpha;
    }

    /**
     * Gets the getModel significance level.
     */
    public double getAlpha() {
        return this.alpha;
    }

    /**
     * @return the list of variables over which this independence checker is capable of determinine independence
     * relations-- that is, all the variables in the given graph or the given data set.
     */
    public List<Node> getVariables() {
        return this.variables;
    }

    /**
     * @return the variable with the given name.
     */
    public Node getVariable(String name) {
        return nameMap.get(name);
    }

    /**
     * @return the list of variable varNames.
     */
    public List<String> getVariableNames() {
        List<Node> variables = getVariables();
        List<String> variableNames = new ArrayList<>();
        for (Node variable1 : variables) {
            variableNames.add(variable1.getName());
        }
        return variableNames;
    }

    /**
     * If <code>isDeterminismAllowed()</code>, deters to IndTestFisherZD; otherwise throws
     * UnsupportedOperationException.
     */
    public boolean determines(List<Node> z, Node x) throws UnsupportedOperationException {
        int[] parents = new int[z.size()];

        for (int j = 0; j < parents.length; j++) {
            parents[j] = covMatrix.getVariables().indexOf(z.get(j));
        }

        int i = covMatrix.getVariables().indexOf(x);

        TetradMatrix matrix2D = covMatrix.getMatrix();
        double variance = matrix2D.get(i, i);

        if (parents.length > 0) {

            // Regress z onto i, yielding regression coefficients b.
            TetradMatrix Czz = matrix2D.getSelection(parents, parents);
            TetradMatrix inverse;

            try {
                inverse = Czz.inverse();
            } catch (Exception e) {
                return true;
            }

            TetradVector Cyz = matrix2D.getColumn(i);
            Cyz = Cyz.viewSelection(parents);
            TetradVector b = inverse.times(Cyz);

            variance -= Cyz.dotProduct(b);
        }

        return variance < 1e-20;
    }

    /**
     * @return the data set being analyzed.
     */
    public DataSet getData() {
        return dataSet;
    }

    public void shuffleVariables() {
        ArrayList<Node> nodes = new ArrayList<>(this.variables);
        Collections.shuffle(nodes);
        this.variables = Collections.unmodifiableList(nodes);
    }

    /**
     * @return a string representation of this test.
     */
    public String toString() {
        return "Fisher's Z, alpha = " + nf.format(getAlpha());
    }

    public void setPValueLogger(PrintStream pValueLogger) {
        this.pValueLogger = pValueLogger;
    }

    //==========================PRIVATE METHODS============================//

    private int sampleSize() {
        return covMatrix().getSampleSize();
    }

    private ICovarianceMatrix covMatrix() {
        return covMatrix;
    }

    private Map<String, Node> mapNames(List<Node> variables) {
        Map<String, Node> nameMap = new ConcurrentHashMap<>();

        for (Node node : variables) {
            nameMap.put(node.getName(), node);
        }

        return nameMap;
    }

    private Map<Node, Integer> indexMap(List<Node> variables) {
        Map<Node, Integer> indexMap = new ConcurrentHashMap<>();

        for (int i = 0; i < variables.size(); i++) {
            indexMap.put(variables.get(i), i);
        }

        return indexMap;
    }

    public void setVariables(List<Node> variables) {
        if (variables.size() != this.variables.size()) throw new IllegalArgumentException("Wrong # of variables.");
        this.variables = new ArrayList<>(variables);
        covMatrix.setVariables(variables);
    }

    public ICovarianceMatrix getCov() {
        return covMatrix;
    }

    @Override
    public List<DataSet> getDataSets() {

        List<DataSet> dataSets = new ArrayList<>();

        dataSets.add(dataSet);

        return dataSets;
    }

    @Override
    public int getSampleSize() {
        return covMatrix.getSampleSize();
    }

    @Override
    public List<TetradMatrix> getCovMatrices() {
        return null;
    }

    @Override
    public double getScore() {
        return -(getPValue() - getAlpha());
    }

    @Override
    public IndependenceTest clone() {
        if (this.dataSet!= null)
            return new IndTestCorrelationT(this.dataSet, this.alpha);
        else
            return new IndTestCorrelationT(this.covMatrix, this.alpha);
    }

    public TDistribution gettDistribution() {
        return tDistribution;
    }

    public boolean isVerbose() {
        return verbose;
    }

    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }
}




