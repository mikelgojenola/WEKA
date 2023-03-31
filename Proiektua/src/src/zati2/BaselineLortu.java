package zati2;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;

public class BaselineLortu {
    public static void main(String[] args) throws Exception {
        // args[0]: train prozesatuaren arff fitxategiaren path-a
        // args[1]: eredua gordetzeko path-a (.model fitxategia)
        // args[2]: kalitatearen neurketa duen txt fitxategia gordetzeko path-a

/*
        String args0 = "/home/mikel/Documentos/DATATA/arff/datuFiltratuak/train.arff";
        String args1 = "/home/mikel/Documentos/DATATA/model/baselineNB.model";
        String args2 = "/home/mikel/Documentos/DATATA/output/baselineOut.txt";
        */

        DataSource source = new DataSource(args[0]);
        Instances train = source.getDataSet();
        train.setClassIndex(train.numAttributes()-1);

        NaiveBayes model = new NaiveBayes();
        model.buildClassifier(train);

        SerializationHelper.write(args[1], model);

        // holdout aplikatu 10 aldiz
        Evaluation eval = new Evaluation(train);

        for(int i = 0; i<10; i++){
            Randomize fRandom = new Randomize();
            fRandom.setRandomSeed(i+1);
            fRandom.setInputFormat(train);
            Instances rData = Filter.useFilter(train, fRandom);

            RemovePercentage fRemove = new RemovePercentage();
            fRemove.setPercentage(66);
            fRemove.setInputFormat(rData);
            Instances testHoldout = Filter.useFilter(rData, fRemove);

            fRemove.setPercentage(66);
            fRemove.setInvertSelection(true);
            fRemove.setInputFormat(rData);
            Instances trainHoldout = Filter.useFilter(rData, fRemove);

            eval.evaluateModel(model, testHoldout);
        }

        FileWriter fw = new FileWriter(new File(args[2]));
        PrintWriter pw = new PrintWriter(fw);
        pw.println(eval.toSummaryString());
        pw.println(eval.toClassDetailsString());
        pw.println(eval.toMatrixString());
        pw.close();
    }
}