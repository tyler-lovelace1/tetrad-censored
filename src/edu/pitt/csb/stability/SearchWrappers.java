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

package edu.pitt.csb.stability;

import edu.cmu.tetrad.data.CovarianceMatrixOnTheFly;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.search.*;
import edu.pitt.csb.mgm.*;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by ajsedgewick on 9/4/15.
 */
public class SearchWrappers {
    public static class PcStableWrapper extends DataGraphSearch {
        //should be one param for the alpha level of the independance test
        public PcStableWrapper(double... params) {
            super(params);
        }

        public PcStableWrapper copy(){return new PcStableWrapper(searchParams);}

        public Graph search(DataSet ds) {
            //System.out.print("Running PCS...");
            IndTestMultinomialAJ indTest = new IndTestMultinomialAJ(ds, searchParams[0],false);
            PcStable pcs = new PcStable(indTest);
            if(initialGraph!=null)
                pcs.setInitialGraph(initialGraph);
            if(knowledge!=null)
                pcs.setKnowledge(knowledge);
            pcs.setDepth((int) searchParams[1]);
            return pcs.search();
        }
    }

    public static class PcMaxWrapper extends DataGraphSearch {
        //should be one param for the alpha level of the independance test
        public PcMaxWrapper(double... params) {
            super(params);
        }

        public PcMaxWrapper copy(){return new PcMaxWrapper(searchParams);}

        public Graph search(DataSet ds) {
            //System.out.print("Running Pc-Max...");
            IndTestMultinomialAJ indTest = new IndTestMultinomialAJ(ds, searchParams[0],true);
            PcMax pcs = new PcMax(indTest);
            if(initialGraph!=null)
                pcs.setInitialGraph(initialGraph);
            if(knowledge!=null)
                pcs.setKnowledge(knowledge);
            pcs.setDepth((int) searchParams[1]);
            return pcs.search();
        }
    }
    public static class CpcStableWrapper extends DataGraphSearch {
        //should be one param for the alpha level of the independance test
        public CpcStableWrapper(double... params) {
            super(params);
        }

        public CpcStableWrapper copy(){return new CpcStableWrapper(searchParams);}

        public Graph search(DataSet ds) {
            IndTestMultinomialAJ indTest = new IndTestMultinomialAJ(ds, searchParams[0],false);
            CpcStable pcs = new CpcStable(indTest);
            if(initialGraph!=null)
                pcs.setInitialGraph(initialGraph);
            if(knowledge!=null)
                pcs.setKnowledge(knowledge);
            pcs.setDepth((int) searchParams[1]);
            return pcs.search();
        }
    }
    public static class Pc50StableWrapper extends DataGraphSearch {
        //should be one param for the alpha level of the independance test
        public Pc50StableWrapper(double... params) {
            super(params);
        }

        public Pc50StableWrapper copy(){return new Pc50StableWrapper(searchParams);}

        public Graph search(DataSet ds) {
            IndTestMultinomialAJ indTest = new IndTestMultinomialAJ(ds, searchParams[0],false);
            Pc50Stable pcs = new Pc50Stable(indTest);
            if(initialGraph!=null)
                pcs.setInitialGraph(initialGraph);
            if(knowledge!=null)
                pcs.setKnowledge(knowledge);
            pcs.setDepth((int) searchParams[1]);
            return pcs.search();
        }
    }
    public static class MFMWrapper extends DataGraphSearch {
        public MFMWrapper(double ... params){super(params);}
        public MFMWrapper copy() {return new MFMWrapper(searchParams);};

        public Graph search(DataSet ds)
        {
            orientations = new HashMap<String,String>();
            double [] lambda = {searchParams[0],searchParams[1],searchParams[2]};
            MGM m = new MGM(ds,lambda);
            Graph g = m.search();
            IndependenceTest i = new IndTestMultinomialAJ(ds,searchParams[3],true);
            FciMaxP f = new FciMaxP(i);
                    f.setInitialGraph(g);
                    Graph g2 =  f.search();
                    for(String x:f.whyOrient.keySet())
                    {
                        orientations.put(x,f.whyOrient.get(x));
                    }
                    return g2;
        }
    }

    public static class MGMFCIWrapper extends DataGraphSearch{
        public MGMFCIWrapper(double...params){super(params);}
        public MGMFCIWrapper copy() {return new MGMFCIWrapper(searchParams);};

        public Graph search(DataSet ds)
        {
            orientations = new HashMap<String,String>();
            double [] lambda = {searchParams[0],searchParams[1],searchParams[2]};
            MGM m = new MGM(ds,lambda);
            Graph g = m.search();
            IndependenceTest i =  new IndTestMultinomialAJ(ds,searchParams[3],true);
            Fci f = new Fci(i);
            f.setInitialGraph(g);
            for(String x:f.whyOrient.keySet())
            {
                orientations.put(x,f.whyOrient.get(x));
            }
            return f.search();

        }
    }

    public static class FCIWrapper extends DataGraphSearch{
        public FCIWrapper(double...params){super(params);}

        public FCIWrapper copy() {return new FCIWrapper(searchParams);};

        public Graph search(DataSet ds)
        {
            //System.out.print("Running FCI...");
            orientations = new HashMap<String,String>();
            IndependenceTest i =  new IndTestMultinomialAJ(ds,searchParams[0],true);
            Fci f = new Fci(i);
            if(initialGraph!=null)
                f.setInitialGraph(initialGraph);
            if(knowledge!=null)
                f.setKnowledge(knowledge);
            f.setDepth((int) searchParams[1]);
            Graph g = f.search();
            Map<String,String> temp = f.whyOrient;
            for(String x:temp.keySet())
            {
                //TODO make sure that this is ok, for concurrency issues in parallel
                orientations.put(x,temp.get(x));
            }
            return g;
        }
    }

    public static class CFCIWrapper extends DataGraphSearch{
        public CFCIWrapper(double...params){super(params);}

        public CFCIWrapper copy() {return new CFCIWrapper(searchParams);};

        public Graph search(DataSet ds)
        {
//            System.out.print("Running CFCI...");
            orientations = new HashMap<>();
            IndependenceTest i =  new IndTestMultinomialCC(ds,searchParams[0],false);
            Cfci f = new Cfci(i);
            // Fixes problems with memory/runtime issues with large, dense graphs
            f.setCompleteRuleSetUsed(false);
            if(initialGraph!=null)
                f.setInitialGraph(initialGraph);
            if(knowledge!=null)
                f.setKnowledge(knowledge);
            f.setDepth((int) searchParams[1]);
            Graph g = f.search();
            Map<String,String> temp = f.whyOrient;
            for(String x:temp.keySet())
            {
                //TODO make sure that this is ok, for concurrency issues in parallel
                orientations.put(x,temp.get(x));
            }
            return g;
        }
    }

    public static class FCI50Wrapper extends DataGraphSearch{
        public FCI50Wrapper(double...params){super(params);}

        public FCI50Wrapper copy() {return new FCI50Wrapper(searchParams);};

        public Graph search(DataSet ds)
        {
            System.out.print("Running FCI50...");
            orientations = new HashMap<>();
            IndependenceTest i =  new IndTestMultinomialCC(ds,searchParams[0],false);
            Fci50 f = new Fci50(i);
            // Fixes problems with memory/runtime issues with large, dense graphs
            f.setCompleteRuleSetUsed(false);
            if(initialGraph!=null)
                f.setInitialGraph(initialGraph);
            if(knowledge!=null)
                f.setKnowledge(knowledge);
            f.setDepth((int) searchParams[1]);
            Graph g = f.search();
            Map<String,String> temp = f.whyOrient;
            for(String x:temp.keySet())
            {
                //TODO make sure that this is ok, for concurrency issues in parallel
                orientations.put(x,temp.get(x));
            }
            return g;
        }
    }

    public static class FCIMAXWrapper extends DataGraphSearch {
        public FCIMAXWrapper(double... params) {
            super(params);
        }

        public FCIMAXWrapper copy() {
            return new FCIMAXWrapper(searchParams);
        }

        public Graph search(DataSet ds) {
            //System.out.print("Running FCI-MAX...");
            orientations = new HashMap<String, String>();
            IndependenceTest i = new IndTestMultinomialCC(ds, searchParams[0],false);
            FciMaxP f = new FciMaxP(i);
            if(initialGraph!=null)
                f.setInitialGraph(initialGraph);
            if(knowledge!=null)
                f.setKnowledge(knowledge);
            Graph g = f.search();
            Map<String, String> temp = f.whyOrient;
            for (String x : temp.keySet()) {
                //TODO make sure that this is ok, for concurrency issues in parallel
                orientations.put(x, temp.get(x));
            }
            return g;

        }
    }

    public static class MGMWrapper extends DataGraphSearch {
        //should be array three parameters for lambdas of each edge type
        public MGMWrapper(double... params) {
            super(params);
        }

        public MGMWrapper copy() {return new MGMWrapper(searchParams);};

        public Graph search(DataSet ds) {
            MGM m = new MGM(ds, searchParams);
            return m.search();
        }
    }
    
    public static class survivalMGMWrapper extends DataGraphSearch {
        //should be array three parameters for lambdas of each edge type
        public survivalMGMWrapper(double... params) {
            super(params);
        }

        public survivalMGMWrapper copy() {return new survivalMGMWrapper(searchParams);};

        public Graph search(DataSet ds) {
        	survivalMGM m = new survivalMGM(ds, searchParams);
            return m.search();
        }
    }

    public static class FgesWrapper extends DataGraphSearch{
        public FgesWrapper(double...params){
            super(params);
        }

        public FgesWrapper copy() {return new FgesWrapper(searchParams);}

        public Graph search(DataSet ds){
            if(ds.isContinuous()) {
                SemBicScore score = new SemBicScore(new CovarianceMatrixOnTheFly(MixedUtils.makeContinuousData(ds)));
                score.setPenaltyDiscount(searchParams[0]);
                Fges fg = new Fges(score);
                if(initialGraph!=null)
                    fg.setInitialGraph(initialGraph);
                if(knowledge!=null)
                    fg.setKnowledge(knowledge);
                return fg.search();
            }
            else if(ds.isDiscrete())
            {
                BDeuScore score = new BDeuScore(ds);
                //score.setSamplePrior(searchParams[0]);
                score.setStructurePrior(searchParams[0]);
                Fges fg = new Fges(score);
                if(initialGraph!=null)
                    fg.setInitialGraph(initialGraph);
                if(knowledge!=null)
                    fg.setKnowledge(knowledge);
                return fg.search();
            }
            else
            {
                ConditionalGaussianScore score = new ConditionalGaussianScore(ds);
                score.setStructurePrior(searchParams[0]);
                Fges fg = new Fges(score);
                if(initialGraph!=null)
                    fg.setInitialGraph(initialGraph);
                if(knowledge!=null)
                    fg.setKnowledge(knowledge);
                return fg.search();
            }

        }
    }

    public static class LiNGWrapper extends DataGraphSearch {
        public LiNGWrapper(double... params) {
            super(params);
        }

        public LiNGWrapper copy() {
            return new LiNGWrapper(searchParams);
        }

        public Graph search(DataSet ds) {
//            System.out.print("Running LiNG...");
            Ling f = new Ling(ds);
            f.setThreshold(searchParams[0]);

            Ling.StoredGraphs gs = f.search();

            System.out.println(gs.getNumGraphs() + " possible graphs found");

            int stableGraphs = 0;
            Graph g = gs.getGraph(0);
            for (int i = 0; i < gs.getNumGraphs(); i++) {
                //TODO make sure that this is ok, for concurrency issues in parallel
                if (gs.isStable(i)) {
                    stableGraphs++;
                    if (stableGraphs==1) {
                        g = gs.getGraph(i);
                    }
                }
            }

            System.out.println(stableGraphs + " stable graphs found");

            return g;

        }
    }
}

