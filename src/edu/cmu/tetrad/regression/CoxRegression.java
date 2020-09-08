package edu.cmu.tetrad.regression;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.jet.math.Functions;
import edu.cmu.tetrad.data.CensoredVariable;
import edu.cmu.tetrad.data.ColtDataSet;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.util.Function;
import edu.cmu.tetrad.util.ProbUtils;
import edu.cmu.tetrad.util.RandomUtil;
import edu.cmu.tetrad.util.TetradSerializable;
import org.apache.commons.lang3.RandomUtils;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.util.MathArrays;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CoxRegression implements TetradSerializable {
    static final long serialVersionUID = 23L;

    private DoubleFactory2D factory2D = DoubleFactory2D.dense;
    private DoubleFactory1D factory1D = DoubleFactory1D.dense;

//    /**
//     *  Matrix of data for the regressors
//     */
//    DoubleMatrix2D X;
//
//    /**
//     *  binary censoring value for target censored variable
//     */
//    int[] censor;
//
//    /**
//     *  order of survival times for target censored variable
//     */
//    int[] order;
//
//    /**
//     *  number of ties for each survival time
//     */
//    int[] H;

    /**
     * The data set that was supplied.
     */
    private DataSet data;

    /**
     * The default alpha level which may be specified otherwise in the GUI
     */
    private double alpha = 0.05;

    private Algebra alg = new Algebra();

    private int[] rows;

    private Graph graph = null;

    //============================CONSTRUCTORS==========================//

    /**
     * Constructs a Cox Proportional Hazards regression model for the given
     * data set.
     *
     * @param data A data set containing censored variables. The regressors
     *             may be continuous or discrete
     */

    public CoxRegression(DataSet data) {
        this.data = data;
        setRows(new int[data.getNumRows()]);
        for (int i = 0; i < getRows().length; i++) getRows()[i] = i;
//        System.out.println("Cox Regression object created");
    }

    /**
     * Generates a simple exemplar of this class to test serialization.
     */
    public static CoxRegression serializableInstance() {
        return new CoxRegression(ColtDataSet.serializableInstance());
    }

    //===========================PUBLIC METHODS========================//

    /**
     * Sets the significance level at which coefficients are judged to be
     * significant.
     *
     * @param alpha the significance level.
     */
    public void setAlpha(double alpha) { this.alpha = alpha; }

    /**
     * @return This graph.
     */
    public Graph getGraph() { return this.graph; }

    /**
     * Regresses <code>target</code> on the <code>regressors</code>, yielding
     * a regression plane.
     *
     * @param target     the target variable, being regressed.
     * @param regressors the list of variables being regressed on.
     * @return the regression plane.
     */
    public CoxRegressionResult regress(CensoredVariable target, List<Node> regressors, int[] _rows) {
//        System.out.println("fitting " + target.getName());

        DoubleMatrix2D X;
        CensoredVariable _target;
        DataSet temp;
        List<Node> tempList = new ArrayList<Node>();
        String[] regressorNames = new String[regressors.size()];

//        System.out.println("Regressors:");
        for (int i = 0; i < regressors.size(); i++) {
            tempList.add(regressors.get(i));
            regressorNames[i] = regressors.get(i).getName();
//            System.out.print(regressorNames[i] + ", ");
        }
//        System.out.println();

//        double[] _time;

        if (regressors.size() == 0) {
            X = factory2D.make(_rows.length, 1, 0.0);
            for (int i = 0; i < _rows.length; i++) {
                X.set(i, 0, RandomUtils.nextDouble(0, 1)-0.5);
            }
            _target = target;
        } else {
            tempList.add(target);
            temp = data.subsetColumns(tempList).subsetRows(_rows);
            _target = (CensoredVariable) temp.getVariable(target.getName());
//            System.out.println(_target.getOrder().length);
//            System.out.println("censor length: " + _target.getCensor().length);
//            System.out.println("order length: " + _target.getOrder().length);
            X = factory2D.make(temp.subsetColumns(regressors).getDoubleData().toArray());

//            if (getRows().length < data.getNumRows()) {
////                _time = temp
//            }
        }

        DoubleMatrix2D hess;
        DoubleMatrix1D beta = factory1D.make(X.columns(), 0.0);
        DoubleMatrix1D grad;
        DoubleMatrix1D p;
        double old_l = Double.POSITIVE_INFINITY;
        double new_l; // = loss(beta, X, target);
        double a, m, t;
        double[] b = new double[X.columns()];
        double[] se = new double[X.columns()];
        double[] z = new double[X.columns()];
        double[] pval = new double[X.columns()];
        double c = 0.1;
//        String[] regressorNames = new String[regressors.size()];

        int Nc = 0;
        for (int i = 0; i < _target.getCensor().length; i++) Nc += _target.getCensor()[i];

        if (regressors.size() == 0) {
            new_l = loss(beta, X, target);
            b[0] = 0.0;
            z[0] = 0.0;
            pval[0] = 1.0;
            se[0] = 1.0;
            return new CoxRegressionResult(regressorNames, _rows.length, b, z, pval, se, 0.5, new_l, this.alpha);
        }

        new_l = loss(beta, X, _target);

//        System.out.println("complete samples: " + _rows.length);

//        System.out.println("everything initialized");

//        if (data.subsetColumns(regressors).isContinuous()) c = 0.5;

//        System.out.println("order: ");
//        for (int idx :  target.getOrder()) System.out.print(idx + " ");
//        System.out.println("H: ");
//        for (int idx :  target.getH()) System.out.print(idx + " ");
//        System.out.println("initial loss: " + new_l);

        while (Math.abs(old_l - new_l) > 1e-5) {
            old_l = new_l;
            a = 1;

            hess = factory2D.make(beta.size(), beta.size(), 0.0);
            grad = factory1D.make(beta.size(), 0.0);

            gradHess(beta, grad, hess, X, _target);

//            System.out.println(grad);
//            System.out.println(hess);

            p = alg.mult(alg.inverse(hess), grad);
            p.assign(Functions.mult(-1.0/Math.sqrt(alg.norm2(p))));

            m = alg.mult(p, grad);

            assert m > 0;

            t = -c * m;

            while (true) {
                new_l = loss(beta.copy().assign(p, Functions.plusMult(a)), X, _target);
//                System.out.println("\t\talpha: " + a);
                if (old_l - new_l <= a*t) {
                    break;
                }
                a /= 2;
            }

            beta.assign(p, Functions.plusMult(a));
//            System.out.println("\tloss: " + new_l);
        }

//        System.out.println("final loss: " + new_l);

        hess = factory2D.make(beta.size(), beta.size(), 0.0);
        grad = factory1D.make(beta.size(), 0.0);

        gradHess(beta, grad, hess, X, _target);

        DoubleMatrix2D cov = alg.inverse(hess);

        int df = X.rows() - regressors.size();
        for (int i = 0; i < regressors.size(); i++) {
            b[i] = beta.get(i);
            se[i] = Math.sqrt(Math.abs(cov.get(i,i)));
            z[i] = b[i] / se[i];
            pval[i] = 2 * (1.0 - ProbUtils.tCdf(Math.abs(z[i]), df)); // norm(z[i]);
            assert !Double.isNaN(pval[i]);
//            System.out.println("b: " + b[i]);
//            System.out.println("var: " + -cov.get(i,i));
//            System.out.println("se: " + se[i]);
//            System.out.println("z: " + z[i]);
//            System.out.println("pval: " + pval[i]);
        }

        return new CoxRegressionResult(regressorNames, X.rows(), b, z, pval, se, 0.5, new_l, this.alpha);
    }

    /**
     * Regresses <code>target</code> on the <code>regressors</code>, yielding
     * a regression plane.
     *
     * @param target     the target variable, being regressed.
     * @param regressors the list of variables being regressed on.
     * @return the regression plane.
     */
    public CoxRegressionResult lassoRegress(CensoredVariable target, List<Node> regressors, int[] _rows, double lambda) {
        System.out.println("fitting " + target.getName());

        DoubleMatrix2D X;
        CensoredVariable _target;
        DataSet temp;
        List<Node> tempList = new ArrayList<Node>();
        String[] regressorNames = new String[regressors.size()];

        System.out.println("Regressors:");
        for (int i = 0; i < regressors.size(); i++) {
            tempList.add(regressors.get(i));
            regressorNames[i] = regressors.get(i).getName();
            System.out.print(regressorNames[i] + ", ");
        }
        System.out.println();

//        double[] _time;

        if (regressors.size() == 0) {
            X = factory2D.make(_rows.length, 1, 0.0);
            for (int i = 0; i < _rows.length; i++) {
                X.set(i, 0, RandomUtils.nextDouble(0, 1)-0.5);
            }
            _target = target;
        } else {
            tempList.add(target);
            temp = data.subsetColumns(tempList).subsetRows(_rows);
            _target = (CensoredVariable) temp.getVariable(target.getName());
//            System.out.println(_target.getOrder().length);
//            System.out.println("censor length: " + _target.getCensor().length);
//            System.out.println("order length: " + _target.getOrder().length);
            X = factory2D.make(temp.subsetColumns(regressors).getDoubleData().toArray());

//            if (getRows().length < data.getNumRows()) {
////                _time = temp
//            }
        }

        DoubleMatrix2D hess;
        DoubleMatrix1D beta = factory1D.make(X.columns(), 0.0);
        DoubleMatrix1D betascale;
        DoubleMatrix1D proxOp;
        DoubleMatrix1D grad;
        DoubleMatrix1D p;
        double old_l = Double.POSITIVE_INFINITY;
        double new_l; // = loss(beta, X, target);
        double a, m, t;
        double[] b = new double[X.columns()];
        double[] se = new double[X.columns()];
        double[] z = new double[X.columns()];
        double[] pval = new double[X.columns()];
        double c = 0.1;
//        String[] regressorNames = new String[regressors.size()];

        int Nc = 0;
        for (int i = 0; i < _target.getCensor().length; i++) Nc += _target.getCensor()[i];

        if (regressors.size() == 0) {
            new_l = loss(beta, X, target);
            b[0] = 0.0;
            z[0] = 0.0;
            pval[0] = 1.0;
            se[0] = 1.0;
            return new CoxRegressionResult(regressorNames, _rows.length, b, z, pval, se, 0.5, new_l, this.alpha);
        }

        new_l = loss(beta, X, _target);

//        System.out.println("complete samples: " + _rows.length);

//        System.out.println("everything initialized");

//        if (data.subsetColumns(regressors).isContinuous()) c = 0.5;

//        System.out.println("order: ");
//        for (int idx :  target.getOrder()) System.out.print(idx + " ");
//        System.out.println("H: ");
//        for (int idx :  target.getH()) System.out.print(idx + " ");
        System.out.println("initial loss: " + new_l);

        while (Math.abs(old_l - new_l) > 1e-5) {
            old_l = new_l;
            a = 1;

            hess = factory2D.make(beta.size(), beta.size(), 0.0);
            grad = factory1D.make(beta.size(), 0.0);

            gradHess(beta, grad, hess, X, _target);

//            System.out.println(grad);
//            System.out.println(hess);

            p = alg.mult(alg.inverse(hess), grad);
            p.assign(Functions.mult(-1.0/Math.sqrt(alg.norm2(p))));

            m = alg.mult(p, grad);

            assert m > 0;

            t = -c * m;

            while (true) {
                new_l = loss(beta.copy().assign(p, Functions.plusMult(a)), X, _target);
//                System.out.println("\t\talpha: " + a);
                if (old_l - new_l <= a*t) {
                    break;
                }
                a /= 2;
            }

            beta.assign(p, Functions.plusMult(a));
            System.out.println("\tloss: " + new_l);
            proxOp = factory1D.make(X.columns(), 1.0);
            betascale = factory1D.make(X.columns(), -a*lambda);
            proxOp.assign(betascale.assign(beta.copy().assign(Functions.abs), Functions.div), Functions.plus);
            proxOp.assign(Functions.max(0.0));
            beta.assign(proxOp, Functions.mult);
//            System.out.println(proxOp);
//            System.out.println(beta);
        }

        System.out.println("final loss: " + new_l);

        hess = factory2D.make(beta.size(), beta.size(), 0.0);
        grad = factory1D.make(beta.size(), 0.0);

        gradHess(beta, grad, hess, X, _target);

        DoubleMatrix2D cov = alg.inverse(hess);

        int df = X.rows() - regressors.size();
        for (int i = 0; i < regressors.size(); i++) {
            b[i] = beta.get(i);
            se[i] = Math.sqrt(Math.abs(cov.get(i,i)));
            z[i] = b[i] / se[i];
            pval[i] = 2 * (1.0 - ProbUtils.tCdf(Math.abs(z[i]), df)); // norm(z[i]);
            assert !Double.isNaN(pval[i]);
//            System.out.println("b: " + b[i]);
//            System.out.println("var: " + -cov.get(i,i));
//            System.out.println("se: " + se[i]);
//            System.out.println("z: " + z[i]);
//            System.out.println("pval: " + pval[i]);
        }

        return new CoxRegressionResult(regressorNames, X.rows(), b, z, pval, se, 0.5, new_l, this.alpha);
    }

    /**
     * Regresses <code>target</code> on the <code>regressors</code>, yielding
     * a regression plane.
     *
     * @param target     the target variable, being regressed.
     * @param regressors the list of variables being regressed on.
     * @return the regression plane.
     */
    public CoxRegressionResult regress(CensoredVariable target, List<Node> regressors, int[] _rows, List<Node> cens, DoubleMatrix2D expVals) {
        System.out.println("fitting " + target.getName());

        DoubleMatrix2D X;
        CensoredVariable _target;
        DataSet temp;
        List<Node> tempList = new ArrayList<Node>();
        String[] regressorNames = new String[regressors.size()];

        System.out.println("Regressors:");
        for (int i = 0; i < regressors.size(); i++) {
            tempList.add(regressors.get(i));
            regressorNames[i] = regressors.get(i).getName();
            System.out.print(regressorNames[i] + ", ");
        }
        System.out.println();

//        double[] _time;

        if (regressors.size() == 0) {
            X = factory2D.make(_rows.length, 1, 0.0);
            for (int i = 0; i < _rows.length; i++) {
                X.set(i, 0, RandomUtils.nextDouble(0, 1)-0.5);
            }
            _target = target;
        } else {
            tempList.add(target);
            temp = data.subsetColumns(tempList).subsetRows(_rows);
            _target = (CensoredVariable) temp.getVariable(target.getName());
//            System.out.println(_target.getOrder().length);
//            System.out.println("censor length: " + _target.getCensor().length);
//            System.out.println("order length: " + _target.getOrder().length);
            X = factory2D.make(temp.subsetColumns(regressors).getDoubleData().toArray());

//            if (getRows().length < data.getNumRows()) {
////                _time = temp
//            }
        }

        int col = 0;
        int idx;
        for (Node _n1 : cens) {
            idx = 0;
            for (Node _n2 : regressors) {
                if (_n1.getName() == _n2.getName()) {
                    X.viewColumn(idx).assign(expVals.viewColumn(col));
                }
                idx++;
            }
            col++;
        }

        DoubleMatrix2D hess;
        DoubleMatrix1D beta = factory1D.make(X.columns(), 0.0);
        DoubleMatrix1D grad;
        DoubleMatrix1D p;
        double old_l = Double.POSITIVE_INFINITY;
        double new_l; // = loss(beta, X, target);
        double a, m, t;
        double[] b = new double[X.columns()];
        double[] se = new double[X.columns()];
        double[] z = new double[X.columns()];
        double[] pval = new double[X.columns()];
        double c = 0.1;
//        String[] regressorNames = new String[regressors.size()];

        int Nc = 0;
        for (int i = 0; i < _target.getCensor().length; i++) Nc += _target.getCensor()[i];

        if (regressors.size() == 0) {
            new_l = loss(beta, X, target);
            b[0] = 0.0;
            z[0] = 0.0;
            pval[0] = 1.0;
            se[0] = 1.0;
            return new CoxRegressionResult(regressorNames, _rows.length, b, z, pval, se, 0.5, new_l, this.alpha);
        }

        new_l = loss(beta, X, _target);

//        System.out.println("complete samples: " + _rows.length);

//        System.out.println("everything initialized");

//        if (data.subsetColumns(regressors).isContinuous()) c = 0.5;

//        System.out.println("order: ");
//        for (int idx :  target.getOrder()) System.out.print(idx + " ");
//        System.out.println("H: ");
//        for (int idx :  target.getH()) System.out.print(idx + " ");
        System.out.println("initial loss: " + new_l);

        while (Math.abs(old_l - new_l) > 1e-5) {
            old_l = new_l;
            a = 1;

            hess = factory2D.make(beta.size(), beta.size(), 0.0);
            grad = factory1D.make(beta.size(), 0.0);

            gradHess(beta, grad, hess, X, _target);

//            System.out.println(grad);
//            System.out.println(hess);

            p = alg.mult(alg.inverse(hess), grad);
            p.assign(Functions.mult(-1.0/Math.sqrt(alg.norm2(p))));

            m = alg.mult(p, grad);

            assert m > 0;

            t = -c * m;

            while (true) {
                new_l = loss(beta.copy().assign(p, Functions.plusMult(a)), X, _target);
//                System.out.println("\t\talpha: " + a);
                if (old_l - new_l <= a*t) {
                    break;
                }
                a /= 2;
            }

            beta.assign(p, Functions.plusMult(a));
            System.out.println("\tloss: " + new_l);
        }

        System.out.println("final loss: " + new_l);

        hess = factory2D.make(beta.size(), beta.size(), 0.0);
        grad = factory1D.make(beta.size(), 0.0);

        gradHess(beta, grad, hess, X, _target);

        DoubleMatrix2D cov = alg.inverse(hess);

        int df = X.rows() - regressors.size();
        for (int i = 0; i < regressors.size(); i++) {
            b[i] = beta.get(i);
            se[i] = Math.sqrt(Math.abs(cov.get(i,i)));
            z[i] = b[i] / se[i];
            pval[i] = 2 * (1.0 - ProbUtils.tCdf(Math.abs(z[i]), df)); // norm(z[i]);
            assert !Double.isNaN(pval[i]);
//            System.out.println("b: " + b[i]);
//            System.out.println("var: " + -cov.get(i,i));
//            System.out.println("se: " + se[i]);
//            System.out.println("z: " + z[i]);
//            System.out.println("pval: " + pval[i]);
        }

        return new CoxRegressionResult(regressorNames, X.rows(), b, z, pval, se, 0.5, new_l, this.alpha);
    }

//    /**
//     * Regresses <code>target</code> on the <code>regressors</code>, yielding
//     * a regression plane.
//     *
//     * @param target     the target variable, being regressed.
//     * @param regressors the list of variables being regressed on.
//     * @return the regression plane.
//     */
//    public CoxRegressionResult regress(CensoredVariable target, Node... regressors, int[] _rows) {
//        List<Node> _regressors = Arrays.asList(regressors);
//        return regress(target, _regressors, _rows);
//    }

    /**
     * The rows in the data used for the regression.
     */
    public void setRows(int[] rows) {
        this.rows = rows;
    }

    //=======================PRIVATE METHODS================================//

    /**
     * The rows in the data used for the regression.
     */
    private int[] getRows() {
        return rows;
    }

    /**
     * Computes the log likelihood of the Cox regression for coefficients beta
     * @param beta the vector of regression coefficients
     * @return the log likelihood of the Cox regression
     */
    private double loss(DoubleMatrix1D beta, DoubleMatrix2D X, CensoredVariable target) {
        int[] order = target.getOrder();
        int[] H = target.getH();
        int[] censor = target.getCensor();
        double loss = 0.0;
        double Hsum, HsumTheta, m, sub;

//        System.out.println("loss function called");

        DoubleMatrix1D theta = alg.mult(X, beta);
        theta.assign(Functions.exp);
        double rs_sum = theta.zSum();

//        System.out.println("Starting loop");

        int i = 0;
        for (int j = 0; j < H.length; j++) {
            Hsum = 0;
            HsumTheta = 0;
            m = 0;
            sub = 0;
            for (int k = 0; k < H[j]; k++) {
                m += censor[order[i+k]];
                Hsum += censor[order[i+k]] * alg.mult(X.viewRow(order[i+k]), beta);
                HsumTheta += censor[order[i+k]] * theta.get(order[i+k]);
                sub += theta.get(order[i+k]);
            }

            assert HsumTheta <= sub;
            assert sub <= rs_sum;

            loss += Hsum;
            for (int l = 0; l < m; l++) {
                loss -= Math.log(rs_sum - ((double) l) / m * HsumTheta);
            }

            i += H[j];
            rs_sum -= sub;
        }

        return loss;
    }

    /**
     * Computes the gradient and the hessian of the Cox regression log likelihood for coefficients beta
     * @param beta the vector of regression coefficients
     * @param grad 0 valued vector that is used to output the gradient
     * @param hess 0 valued matrix that is used to output the hessian
     */
    private void gradHess(DoubleMatrix1D beta, DoubleMatrix1D grad, DoubleMatrix2D hess, DoubleMatrix2D X, CensoredVariable target) {
        int[] order = target.getOrder();
        int[] H = target.getH();
        int[] censor = target.getCensor();
        double HsumTheta, m, sub, d, phi;

        DoubleMatrix1D theta = alg.mult(X, beta);
        theta.assign(Functions.exp);
        double rs_sum = theta.zSum();

        DoubleMatrix2D temp = factory2D.make(beta.size(), beta.size(), 0.0);
        DoubleMatrix2D outer_num = factory2D.make(beta.size(), beta.size(), 0.0);
        DoubleMatrix1D num = factory1D.make(beta.size(), 0.0);
        DoubleMatrix1D sub_num, HsumVec, HsumThetaVec, Z;
        DoubleMatrix2D sub_outer, HsumOuter;

        for (int i = 0; i < X.rows(); i++) {
            num.assign(X.viewRow(i), Functions.plusMult(theta.get(i)));
//            temp = factory2D.make(beta.size(), beta.size(), 0.0);
            alg.multOuter(X.viewRow(i), X.viewRow(i), temp);
            outer_num.assign(temp, Functions.plusMult(theta.get(i)));
        }

        int i = 0;
        for (int j = 0; j < H.length; j++) {
            HsumTheta = 0;
            m = 0;
            sub = 0;

            HsumVec = factory1D.make(beta.size(), 0.0);
            HsumThetaVec = factory1D.make(beta.size(), 0.0);
            sub_num = factory1D.make(beta.size(), 0.0);

            HsumOuter = factory2D.make(beta.size(), beta.size(), 0.0);
            sub_outer = factory2D.make(beta.size(), beta.size(), 0.0);

            for (int k = 0; k < H[j]; k++) {
                m += censor[order[i+k]];
                sub += theta.get(order[i+k]);

//                temp = factory2D.make(beta.size(), beta.size(), 0.0);
                alg.multOuter(X.viewRow(order[i+k]), X.viewRow(order[i+k]), temp);
                temp.assign(Functions.mult(theta.get(order[i+k])));

                HsumTheta += censor[order[i+k]] * theta.get(order[i+k]);
                HsumVec.assign(X.viewRow(order[i+k]), Functions.plusMult(censor[order[i+k]]));
                HsumThetaVec.assign(X.viewRow(order[i+k]), Functions.plusMult(censor[order[i+k]]*theta.get(order[i+k])));
                HsumOuter.assign(temp, Functions.plusMult(censor[order[i+k]]));

                sub_num.assign(X.viewRow(order[i+k]), Functions.plusMult(theta.get(order[i+k])));
                sub_outer.assign(temp, Functions.plus);
            }

            assert HsumTheta <= sub;
            assert sub <= rs_sum;

            grad.assign(HsumVec, Functions.plus);
            for (int l = 0; l < m; l++) {
                d = ((double) l) / m;
                Z = num.copy().assign(HsumThetaVec, Functions.minusMult(d));
                phi = rs_sum - d * HsumTheta;
                grad.assign(Z, Functions.minusMult(1/phi));
//                temp = factory2D.make(beta.size(), beta.size(), 0.0);
                alg.multOuter(Z, Z, temp);
                hess.assign(outer_num.copy().assign(HsumOuter, Functions.minusMult(d)), Functions.minusMult(1/phi));
                hess.assign(temp, Functions.plusMult(1/(phi*phi)));
            }

//            System.out.println("H_j = " + H[j]);
//            System.out.println(outer_num);
//            System.out.println(HsumOuter);

            i += H[j];
            rs_sum -= sub;
            num.assign(sub_num, Functions.minus);
            outer_num.assign(sub_outer, Functions.minus);
        }

    }

    /**
     * Calculates p value from Z score of a regression coefficient
     * @param z the Z score of a regression coefficient
     * @return the p value
     */
    private double norm(double z) {
        assert !Double.isNaN(z);
        NormalDistribution nd = new NormalDistribution();
        double p = 2 * nd.cumulativeProbability(-Math.abs(z));
        assert !Double.isNaN(p);
        return p;
    }

}
