package zati3;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.FileWriter;
import java.io.PrintWriter;


public class Sailkapena {
    public static void main(String[] args) throws Exception {
        // args[0]: test_blind datuen arff fitxategiaren path-a
        // args[1]: ereduaren model fitxategiaren path-a
        // args[2]: irteerako datuak gordetzeko txt fitxategiaren path-a


        DataSource source = new DataSource(args[0]);
        Instances testBlind = source.getDataSet();
        testBlind.setClassIndex(testBlind.numAttributes()-1);

        MultilayerPerceptron model = (MultilayerPerceptron) SerializationHelper.read(args[1]);

        Evaluation eval = new Evaluation(testBlind);
        eval.evaluateModel(model, testBlind);

        FileWriter fw = new FileWriter(args[2]);

        PrintWriter pw = new PrintWriter(fw);
        pw.println("");
        pw.println(eval.toSummaryString());
        pw.println(eval.toClassDetailsString());
        pw.println(eval.toMatrixString());
        pw.close();
    }
}