package edu.cmu.tetrad.algcomparison.algorithm.oracle.pag;

import edu.cmu.tetrad.algcomparison.algorithm.Algorithm;
import edu.cmu.tetrad.algcomparison.independence.IndependenceWrapper;
import edu.cmu.tetrad.algcomparison.score.ScoreWrapper;
import edu.cmu.tetrad.algcomparison.utils.HasKnowledge;
import edu.cmu.tetrad.algcomparison.utils.TakesInitialGraph;
import edu.cmu.tetrad.data.DataModel;
import edu.cmu.tetrad.data.DataType;
import edu.cmu.tetrad.data.IKnowledge;
import edu.cmu.tetrad.graph.EdgeListGraph;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.search.TsDagToPag;
import edu.cmu.tetrad.util.Parameters;

import java.util.List;

/**
 * tsFCI.
 *
 * @author jdramsey
 * @author Daniel Malinsky
 */
public class TsGfci implements Algorithm, TakesInitialGraph, HasKnowledge {
    static final long serialVersionUID = 23L;
    private IndependenceWrapper test;
    private ScoreWrapper score;
    private Algorithm initialGraph = null;
    private IKnowledge knowledge = null;

    public TsGfci(IndependenceWrapper type, ScoreWrapper score) {
        this.test = type;
        this.score = score;
    }

    @Override
    public Graph search(DataModel dataSet, Parameters parameters) {
        edu.cmu.tetrad.search.TsGFci search = new edu.cmu.tetrad.search.TsGFci(test.getTest(dataSet, parameters),
                score.getScore(dataSet, parameters));
        search.setKnowledge(dataSet.getKnowledge());
        return search.search();
    }

    @Override
    public Graph getComparisonGraph(Graph graph) { return new TsDagToPag(new EdgeListGraph(graph)).convert(); }

    public String getDescription() {
        return "tsGFCI (Time Series GFCI) using " + test.getDescription() + " and " + score.getDescription() +
                (initialGraph != null ? " with initial graph from " +
                        initialGraph.getDescription() : "");
    }

    @Override
    public DataType getDataType() {
        return test.getDataType();
    }

    @Override
    public List<String> getParameters() {
        List<String> parameters = test.getParameters();
        parameters.addAll(score.getParameters());
        parameters.add("faithfulnessAssumed");
        parameters.add("maxIndegree");
        parameters.add("printStream");
        return parameters;
    }

    @Override
    public IKnowledge getKnowledge() {
        return this.knowledge;
    }

    @Override
    public void setKnowledge(IKnowledge knowledge) {
        this.knowledge = knowledge;
    }
}
