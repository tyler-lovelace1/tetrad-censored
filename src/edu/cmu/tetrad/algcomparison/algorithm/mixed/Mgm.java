package edu.cmu.tetrad.algcomparison.algorithm.mixed;

import edu.cmu.tetrad.algcomparison.algorithm.Algorithm;
import edu.cmu.tetrad.data.DataModel;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DataType;
import edu.cmu.tetrad.data.DataUtils;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.GraphUtils;
import edu.cmu.tetrad.util.Parameters;
import edu.pitt.csb.mgm.MGM;

import java.util.ArrayList;
import java.util.List;

/**
 * @author jdramsey
 */
public class Mgm implements Algorithm {
    static final long serialVersionUID = 23L;
    public Graph search(DataModel ds, Parameters parameters) {
        DataSet _ds = DataUtils.getMixedDataSet(ds);
        MGM m = new MGM(_ds, new double[]{
                parameters.getDouble("mgmParam1"),
                parameters.getDouble("mgmParam2"),
                parameters.getDouble("mgmParam3")
        });
        return m.search();
    }

    // Need to marry the parents on this.
    public Graph getComparisonGraph(Graph graph) {
        return GraphUtils.undirectedGraph(graph);
    }

    public String getDescription() {
        return "Returns the output of the MGM (Mixed Graphical Model) algorithm (a Markov random field)";
    }

    @Override
    public DataType getDataType() {
        return DataType.Mixed;
    }

    @Override
    public List<String> getParameters() {
        List<String> params = new ArrayList<>();
        params.add("mgmParam1");
        params.add("mgmParam2");
        params.add("mgmParam3");
        return params;
    }
}