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
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.linalg.Algebra;
import cern.jet.math.Functions;

import java.io.FileWriter;
import java.io.PrintWriter;

/**
 * Implementation of Nesterov's 83 method as described in Beck and Teboulle, 2009
 * aka Fast Iterative Shrinkage Thresholding Algorithm
 *
 * with step size scaling from Becker et all 2011
 *
 * Created by ajsedgewick on 7/29/15.
 */
public class ProximalGradient {

    // Factors to alter Lipshitz constant estimate L, used for stepsize t = 1/L
    private double beta; //factor to increase L when Lipshitz violated
    private double alpha; //factor to decrease L otherwise
    private double L; //value of Lipshitz constant estimate L
    private Algebra alg = new Algebra();
    private DoubleFactory1D factory1D = DoubleFactory1D.dense;

    private boolean edgeConverge; //if this is true we look to stop optimization when the edge predictions stop changing
    private int noEdgeChangeTol = 3; //number of iterations in a row with no edge changes before we break

    private int printIter = 100;
    private double backtrackTol = 1e-10;
    public int iterComplete = 0;
    public double timePerIter = 0;


    /**
     * Constructor, set parameters for a proximal gradient run
     *
     * @param beta (0,1) factor to increase L when Lipshitz violated, L = L_old/beta
     * @param alpha (0,1) factor to decrease L otherwise, L = L_old*alpha
     * @param edgeConverge
     */
    public ProximalGradient(double beta, double alpha, boolean edgeConverge){
        if(beta <= 0 || beta >=1)
            throw new IllegalArgumentException("beta must be (0,1): " + beta);

        if(alpha <= 0 || alpha >=1)
            throw new IllegalArgumentException("alpha must be (0,1): " + alpha);

        this.L = 1.0;
        this.beta = beta;
        this.alpha = alpha;
        this.edgeConverge = edgeConverge;
    }

    /**
     * Constructor using defaults from Becker et al 2011. beta = .5, alpha = .9
     */
    public ProximalGradient(){
        beta = .5;
        alpha = .9;
        edgeConverge = false;
        this.L = 1.0;
    }

    /**
     *  Positive edge change tolerance is the number of iterations with 0 edge changes needed to converge.
     *  Negative edge change tolerance means convergence happens when number of difference edges <= |edge change tol|.
     *  Default is 3.
     */
    public void setEdgeChangeTol(int t){
        noEdgeChangeTol = t;
    }



    //run FISTA with step size backtracking attempt to speed up
    public DoubleMatrix1D learnBackTrack(ConvexProximal cp, DoubleMatrix1D Xin, double epsilon, int iterLimit,long time) {
        long start = System.nanoTime();
        DoubleMatrix1D X = cp.proximalOperator(1.0, Xin.copy());
        DoubleMatrix1D Y = X.copy();
        DoubleMatrix1D Z = X.copy();
        DoubleMatrix1D GrY = cp.smoothGradient(Y);
        DoubleMatrix1D GrX = cp.smoothGradient(X);

        int iterCount = 0;
        int noEdgeChangeCount = 0;

        double theta = Double.POSITIVE_INFINITY;
        double thetaOld = theta;
//        L = 1.0;
        double Lold ;


        boolean backtrackSwitch = true;
        double dx;
        double Fx = Double.POSITIVE_INFINITY;
        double Gx = Double.POSITIVE_INFINITY;
        double Fy;
        double obj;

        while (true) {
            long lastStart = System.nanoTime();
            Lold = L;
            L = L*alpha;
            thetaOld = theta;
            DoubleMatrix1D Xold = X.copy();
            obj = Fx + Gx;
            while(true) {
                theta = 2.0/(1.0+Math.sqrt(1.0+(4.0*L)/(Lold*Math.pow(thetaOld,2))));

                if(theta < 1){
                    Y.assign(Xold.copy().assign(Functions.mult(1 - theta)));
                    Y.assign(Z.copy().assign(Functions.mult(theta)), Functions.plus);
                }

                Fy = cp.smooth(Y, GrY);
                DoubleMatrix1D temp = Y.copy().assign(GrY.copy().assign(Functions.mult(1.0 / L)), Functions.minus);
                Gx = cp.nonSmooth(1.0 / L, temp, X);
                if(backtrackSwitch){
                    Fx = cp.smoothValue(X);
                } else {
                    //tempPar = new MGMParams();
                    Fx = cp.smooth(X, GrX);
                    //GrX.assign(factory1D.make(tempPar.toVector()[0]));
                }

                DoubleMatrix1D XmY = X.copy().assign(Y, Functions.minus);
                double normXY = alg.norm2(XmY);
                if(normXY==0)
                    break;

                double Qx;
                double LocalL;

                if(backtrackSwitch){
                    //System.out.println("Back Norm");
                    Qx = Fy + alg.mult(XmY, GrY) + (L / 2.0) * normXY;
                    LocalL = L + 2*Math.max(Fx - Qx, 0)/normXY;
                    backtrackSwitch =  Math.abs(Fy - Fx) >= backtrackTol * Math.max(Math.abs(Fx), Math.abs(Fy));
                } else {
                    //System.out.println("Close Rule");

                    //it shouldn't be possible for GrX to be null here...
                    //if(GrX==null)
                    //GrX = factory1D.make(gradient(Xpar).toVector()[0]);
                    //Fx = alg.mult(YmX, Gx.assign(G, Functions.minus));
                    //Qx = (L / 2.0) * alg.norm2(YmX);
                    LocalL = 2*alg.mult(XmY, GrX.assign(GrY, Functions.minus))/normXY;

                }
                //if(-1e-8 <= Qx - Fx){
                //if(Fx <= Qx){
                //System.out.println("LocalL: " + LocalL + " L: " + L);
                if(LocalL <= L){
                    break;
                } else if (LocalL != Double.POSITIVE_INFINITY) {
                    L = LocalL;
                } else {
                    LocalL = L;
                }

                L = Math.max(LocalL, L/beta);

            }

            int diffEdges = 0;
            for(int i =0; i<X.size(); i++){
                double a = X.get(i);
                double b = Xold.get(i);
                if(a!=0 &  b==0){
                    diffEdges++;
                } else if (a==0 & b!=0){
                    diffEdges++;
                }
            }

            dx = norm2(X.copy().assign(Xold, Functions.minus)) / Math.max(1,norm2(X));

            //sometimes there are more edge changes after initial 0, so may want to do two zeros in a row...
            if (diffEdges == 0 && edgeConverge) {
                noEdgeChangeCount++;
                if(noEdgeChangeCount >= noEdgeChangeTol) {
                    //    System.out.println("Edges converged at iter: " + iterCount + " with |dx|/|x|: " + dx);
                    //    System.out.println("Iter: " + iterCount + " |dx|/|x|: " + dx + " normX: " + norm2(X) + " nll: " +
                    //                     Fx + " reg: " + Gx + " DiffEdges: " + diffEdges + " L: " + L);
                    break;
                }
                // negative noEdgeChangeTol stops when diffEdges <= |noEdgeChangeTol|
            } else if (noEdgeChangeTol < 0 && diffEdges <= Math.abs(noEdgeChangeTol)) {
                //   System.out.println("Edges converged at iter: " + iterCount + " with |dx|/|x|: " + dx);
                //    System.out.println("Iter: " + iterCount + " |dx|/|x|: " + dx + " normX: " + norm2(X) + " nll: " +
                //                Fx + " reg: " + Gx + " DiffEdges: " + diffEdges + " L: " + L);
                break;
            } else {
                noEdgeChangeCount = 0;
            }

            //edge converge should happen before params converge, unless epsilon is big
            if (dx < epsilon && !edgeConverge) {
                //       System.out.println("Converged at iter: " + iterCount + " with |dx|/|x|: " + dx + " < epsilon: " + epsilon);
                //      System.out.println("Iter: " + iterCount + " |dx|/|x|: " + dx + " normX: " + norm2(X) + " nll: " +
                //              Fx + " reg: " + Gx + " DiffEdges: " + diffEdges + " L: " + L);
                break;
            }

            //restart acceleration if objective got worse
            if(Fx + Gx > obj) {
                theta = Double.POSITIVE_INFINITY;
                Y.assign(X.copy());
                //Ypar = new MGMParams(Xpar);
                Z.assign(X.copy());
                //Fy = Fx;
                //GrY.assign(GrX.copy());
            }else if(theta==1){
                Z.assign(X.copy());
            } else {
                Z.assign(X.copy().assign(Functions.mult(1 / theta)));
                Z.assign(Xold.copy().assign(Functions.mult(1 - (1.0 / theta))), Functions.plus);
            }


            printIter = 1;
            if (iterCount % printIter == 0) {
                //System.out.println("Iter: " + iterCount + " |dx|/|x|: " + dx + " normX: " + norm2(X) + " nll: " + Fx + " reg: " + Gx + " DiffEdges: " + diffEdges + " L: " + L);
                //  System.out.println("Iter: " + iterCount + " |dx|/|x|: " + dx + " nll: " + negLogLikelihood(params) + " reg: " + regTerm(params));
            }
            //System.out.println("t: " + t);
            //System.out.println("Params: " + params);

            iterCount++;
            if (iterCount >= iterLimit) {
                //         System.out.println("Iter limit reached");
                break;
            }


            timePerIter+= (System.nanoTime()-lastStart)/Math.pow(10,9);
            iterComplete++;
            if((System.nanoTime()-start)>=time) {
                return X;
            }
        }
        return X;
    }


    //run FISTA with step size backtracking attempt to speed up
    public DoubleMatrix1D learnBackTrack(ConvexProximal cp, DoubleMatrix1D Xin, double epsilon, int iterLimit) {
//        System.out.println("intial L: " + L);
        DoubleMatrix1D X = Xin.copy(); // cp.proximalOperator(1.0 / L, Xin.copy());
        DoubleMatrix1D Y = X.copy();
        DoubleMatrix1D Z = X.copy();
        DoubleMatrix1D GrY = cp.smoothGradient(Y);
        DoubleMatrix1D GrX = cp.smoothGradient(X);

        int iterCount = 0;
        int noEdgeChangeCount = 0;

        double theta = Double.POSITIVE_INFINITY;
        double thetaOld;
//        L = 1.0;
        double Lold;


        boolean backtrackSwitch = true;
        double dx;
        double Fx = Double.POSITIVE_INFINITY;
        double Gx = Double.POSITIVE_INFINITY;
        double Fy;
        double obj;

        while (true) {
            Lold = L;
            L = L*alpha;
            thetaOld = theta;
            DoubleMatrix1D Xold = X.copy();
            obj = Fx + Gx;
            while(true) {
                if (thetaOld != Double.POSITIVE_INFINITY)
                    theta = 2.0/(1.0+Math.sqrt(1.0+(4.0*L)/(Lold*Math.pow(thetaOld,2))));
                else
                    theta = 1.0;

//                System.out.println("theta: " + theta);

//                if(Double.isNaN(theta)) {
//                    System.out.println("Lold: " + Lold);
//                    System.out.println("L: " + L);
//                    System.out.println("thetaOld: " + thetaOld);
//                    System.out.println(Lold*Math.pow(thetaOld,2));
//                    System.out.println((4.0*L)/(Lold*Math.pow(thetaOld,2)));
//                    System.out.println(Math.sqrt(1.0+(4.0*L)/(Lold*Math.pow(thetaOld,2))));
////                    System.out.println("X: " + X);
////                    System.out.println("Y: " + Y);
////                    System.out.println("X-Y: " + XmY);
//                    // break;
//                }

                if(theta < 1){
                    Y.assign(Xold.copy().assign(Functions.mult(1 - theta)));
                    Y.assign(Z.copy().assign(Functions.mult(theta)), Functions.plus);
                }

                Fy = cp.smooth(Y, GrY);
                DoubleMatrix1D temp = Y.copy().assign(GrY.copy().assign(Functions.mult(1.0 / L)), Functions.minus);
                Gx = cp.nonSmooth(1.0 / L, temp, X);
                if(backtrackSwitch){
                    Fx = cp.smoothValue(X);
                } else {
                    //tempPar = new MGMParams();
                    Fx = cp.smooth(X, GrX);
                    //GrX.assign(factory1D.make(tempPar.toVector()[0]));
                }

                DoubleMatrix1D XmY = X.copy().assign(Y, Functions.minus);
                double normXY = alg.norm2(XmY);
                if(normXY==0)
                    break;

//                System.out.println("X: " + X);
//                System.out.println("Y: " + Y);
//                System.out.println("XmY: " + XmY);
//                System.out.println("normXY: " + normXY);

                double Qx;
                double LocalL;

                if(backtrackSwitch){
//                    System.out.println("            Back Norm");
                    Qx = Fy + alg.mult(XmY, GrY) + (L / 2.0) * normXY;
                    LocalL = L + 2*Math.max(Fx - Qx, 0)/normXY;
                    backtrackSwitch =  Math.abs(Fy - Fx) >= backtrackTol * Math.max(Math.abs(Fx), Math.abs(Fy));
//                    System.out.println("LocalL: " + LocalL);
                } else {
                    // System.out.println("            Close Rule: " + normXY);

                    //it shouldn't be possible for GrX to be null here...
                    //if(GrX==null)
                    //GrX = factory1D.make(gradient(Xpar).toVector()[0]);
                    //Fx = alg.mult(YmX, Gx.assign(G, Functions.minus));
                    //Qx = (L / 2.0) * alg.norm2(YmX);
                    LocalL = 2*alg.mult(XmY, GrX.assign(GrY, Functions.minus))/normXY;

                }
                 //if(-1e-8 <= Qx - Fx){
                //if(Fx <= Qx){
//                if (LocalL == Double.POSITIVE_INFINITY) {
//                System.out.println("            LocalL: " + LocalL + " L: " + L);
//                }

                if (Double.isNaN(LocalL)) {
                    throw new RuntimeException("LocalL is NaN");
//
//                    System.out.println("LocalL: " + LocalL);
//                    System.out.println("Lold: " + Lold);
//                    System.out.println("L: " + L);
//                    System.out.println("thetaOld: " + thetaOld);
//                    System.out.println("theta: " + theta);
//                    System.out.println("normXY: " + normXY);
//                    System.out.println("backtrackSwitch: " + backtrackSwitch);
////                    System.out.println("X-Y: " + XmY);
////                    System.out.println("GrX-GrY: " + GrX.assign(GrY, Functions.minus));
//                    System.out.println("(X-Y) @ (GrX-GrY): " + alg.mult(XmY, GrX.copy().assign(GrY, Functions.minus)));
//                    System.out.println("(X-Y)/normXY @ (GrX-GrY)/normXY: " + alg.mult(XmY.copy().assign(Functions.div(normXY)), GrX.copy().assign(GrY, Functions.minus).assign(Functions.div(normXY))));
//                    System.out.println("2 * (X-Y) @ (GrX-GrY) / normXY: " + 2 * alg.mult(XmY, GrX.copy().assign(GrY, Functions.minus)) / normXY);
//                    LocalL = 2 * alg.mult(XmY, GrX.assign(GrY, Functions.minus)) / normXY;
//                    System.out.println("LocalL: " + LocalL);

//                    if (Double.isNaN(LocalL)) throw new RuntimeException("LocalL is NaN");
                }
////                    throw new Exception("LocalL is NaN");
////                    break;
//
//                    if (Double.isNaN(LocalL)) {
//                        FileWriter fileWriter = new FileWriter("errorfile");
//                        fileWriter.write("LocalL: " + LocalL);
//                        fileWriter.write("Lold: " + Lold);
//                        fileWriter.write("L: " + L);
//                        fileWriter.write("thetaOld: " + thetaOld);
//                        fileWriter.write("theta: " + theta);
//                        fileWriter.write("normXY: " + normXY);
//                        fileWriter.write("backtrackSwitch: " + backtrackSwitch);
//                        fileWriter.write("X-Y: " + XmY);
//                        fileWriter.write("GrX-GrY: " + GrX.assign(GrY, Functions.minus));
//                        fileWriter.write("(X-Y) @ (GrX-GrY): " + alg.mult(XmY, GrX.assign(GrY, Functions.minus)));
//                        fileWriter.write("2 * (X-Y) @ (GrX-GrY) / normXY: " + 2 * alg.mult(XmY, GrX.assign(GrY, Functions.minus)) / normXY);
//                        LocalL = 2 * alg.mult(XmY, GrX.assign(GrY, Functions.minus)) / normXY;
//                        fileWriter.write("LocalL: " + LocalL);
//                        fileWriter.close();
//
//                        throw new Exception("LocalL is NaN");
//                    }
//                }

//                System.out.println("LocalL: " + LocalL);

                if(LocalL <= L){
                    break;
                } else if (LocalL != Double.POSITIVE_INFINITY) {
                    L = LocalL;
                } else {
                    LocalL = L;
                }

                L = Math.max(LocalL, L/beta);
                
                //System.out.println("L: " + L);

            }

            int diffEdges = 0;
            for(int i =0; i<X.size(); i++){
                double a = X.get(i);
                double b = Xold.get(i);
                if(a!=0 &  b==0){
                    diffEdges++;
                } else if (a==0 & b!=0){
                    diffEdges++;
                }
            }

            dx = norm2(X.copy().assign(Xold, Functions.minus)) / Math.max(1,norm2(X));

//            System.out.println(noEdgeChangeCount + ", " + diffEdges + ", " + dx);

            //sometimes there are more edge changes after initial 0, so may want to do two zeros in a row...
            if (diffEdges == 0 && edgeConverge) {
                noEdgeChangeCount++;
                if(noEdgeChangeCount >= noEdgeChangeTol) {
                //    System.out.println("Edges converged at iter: " + iterCount + " with |dx|/|x|: " + dx);
                //    System.out.println("Iter: " + iterCount + " |dx|/|x|: " + dx + " normX: " + norm2(X) + " nll: " +
       //                     Fx + " reg: " + Gx + " DiffEdges: " + diffEdges + " L: " + L);
                    break;
                }
                // negative noEdgeChangeTol stops when diffEdges <= |noEdgeChangeTol|
            } else if (noEdgeChangeTol < 0 && diffEdges <= Math.abs(noEdgeChangeTol)) {
             //   System.out.println("Edges converged at iter: " + iterCount + " with |dx|/|x|: " + dx);
            //    System.out.println("Iter: " + iterCount + " |dx|/|x|: " + dx + " normX: " + norm2(X) + " nll: " +
        //                Fx + " reg: " + Gx + " DiffEdges: " + diffEdges + " L: " + L);
                break;
            } else {
                noEdgeChangeCount = 0;
            }

            //edge converge should happen before params converge, unless epsilon is big
            if (dx < epsilon && !edgeConverge) {
         //       System.out.println("Converged at iter: " + iterCount + " with |dx|/|x|: " + dx + " < epsilon: " + epsilon);
          //      System.out.println("Iter: " + iterCount + " |dx|/|x|: " + dx + " normX: " + norm2(X) + " nll: " +
          //              Fx + " reg: " + Gx + " DiffEdges: " + diffEdges + " L: " + L);
                break;
            }

            //restart acceleration if objective got worse
            if(Fx + Gx > obj) {
                theta = Double.POSITIVE_INFINITY;
                Y.assign(X.copy());
                //Ypar = new MGMParams(Xpar);
                Z.assign(X.copy());
                //Fy = Fx;
                //GrY.assign(GrX.copy());
            }else if(theta==1){
                Z.assign(X.copy());
            } else {
                Z.assign(X.copy().assign(Functions.mult(1 / theta)));
                Z.assign(Xold.copy().assign(Functions.mult(1 - (1.0 / theta))), Functions.plus);
            }


            printIter = 50;
            if (iterCount % printIter == 0) {
                //System.out.println("Iter: " + iterCount + " |dx|/|x|: " + dx + " normX: " + norm2(X) + " nll: " + Fx + " reg: " + Gx + " DiffEdges: " + diffEdges + " L: " + L);
              //  System.out.println("Iter: " + iterCount + " |dx|/|x|: " + dx + " nll: " + negLogLikelihood(params) + " reg: " + regTerm(params));
                double temp = Fx+Gx;
                System.out.println("        " + Thread.currentThread().getName() + ": Iter  " + iterCount + ": " + temp);
            }
            //System.out.println("t: " + t);
            //System.out.println("Params: " + params);

            iterCount++;
            if (iterCount >= iterLimit) {
       //         System.out.println("Iter limit reached");
                break;
            }
        }
        double temp = Fx+Gx;
        System.out.println("    " + Thread.currentThread().getName() + ": Finished, iter: " + iterCount + ": " + temp);
        return X;
    }

    public static double norm2(DoubleMatrix1D vec){
        //return Math.sqrt(vec.copy().assign(Functions.pow(2)).zSum());
        return Math.sqrt(new Algebra().norm2(vec));
    }
}

