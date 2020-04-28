package edu.cmu.tetrad.algcomparison.independence;

import edu.cmu.tetrad.data.DataModel;
import edu.cmu.tetrad.data.DataType;
import edu.cmu.tetrad.data.DataUtils;
import edu.cmu.tetrad.search.BicScore;
import edu.cmu.tetrad.search.IndTestScore;
import edu.cmu.tetrad.search.IndependenceTest;
import edu.cmu.tetrad.search.Score;
import edu.cmu.tetrad.util.Parameters;

import java.util.ArrayList;
import java.util.List;

/**
 * Wrapper for Fisher Z test.
 *
 * @author jdramsey
 */
public class DiscreteBicTest implements IndependenceWrapper {
    static final long serialVersionUID = 23L;

    @Override
    public IndependenceTest getTest(DataModel dataSet, Parameters parameters) {
        Score score = new BicScore(DataUtils.getDiscreteDataSet(dataSet));
//        score.setSamplePrior(parameters.getDouble("samplePrior"));
//        score.setStructurePrior(parameters.getDouble("structurePrior"));
        return new IndTestScore(score);
    }

    @Override
    public String getDescription() {
        return "SEM BIC test";
    }

    @Override
    public DataType getDataType() {
        return DataType.Continuous;
    }

    @Override
    public List<String> getParameters() {
        List<String> parameters = new ArrayList<>();
//        parameters.add("samplePrior");
//        parameters.add("structurePrior");
        return parameters;
    }
}
