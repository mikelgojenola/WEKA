package zati2;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;

public class EreduOptimoaLortu {
    public static void main(String[] args) throws Exception {
        // args[0]: train filtratuaren fitxategiaren path-a
        // args[1]: eredu optimoa gordetzeko path-a
        // args[2]: kalitatearen neurketa duen txt fitxategia gordetzeko path-a

/*
        String args0 = "/home/mikel/Documentos/DATATA/arff/datuFiltratuak/train.arff";
        String args1 = "/home/mikel/Documentos/DATATA/model/MLP.model";
        String args2 = "/home/mikel/Documentos/DATATA/output/MLPout.txt";
        */

        DataSource source = new DataSource(args[0]);
        Instances train = source.getDataSet();
        train.setClassIndex(train.numAttributes()-1);

        Instances dataMini = datuakTxikitu(train, 0.0);

        MultilayerPerceptron model = new MultilayerPerceptron();

        String hiddenLayers = "70";

        model.setHiddenLayers(hiddenLayers);
        model.setLearningRate(0.05);
        model.setTrainingTime(150);

        model.buildClassifier(dataMini);

        SerializationHelper.write(args[1], model);

        Evaluation evalEzZintzoa = new Evaluation(train);
        evalEzZintzoa.evaluateModel(model, dataMini);

        FileWriter fw = new FileWriter(new File(args[2]));
        PrintWriter pw = new PrintWriter(fw);

        //
        pw.println("-------------------- EBALUAZIO EZ-ZINTZOA --------------------");
        pw.println(evalEzZintzoa.toSummaryString());
        pw.println(evalEzZintzoa.toClassDetailsString());
        pw.println(evalEzZintzoa.toMatrixString());
        pw.println("");

        // holdout aplikatu 10 aldiz
        Evaluation evalHO = new Evaluation(dataMini);

        for(int i = 0; i<50; i++){
            System.out.println("loop: " + i);
            Randomize fRandom = new Randomize();
            fRandom.setRandomSeed(i+1);
            fRandom.setInputFormat(dataMini);
            weka.core.Instances rData = Filter.useFilter(train, fRandom);

            RemovePercentage fRemove = new RemovePercentage();
            fRemove.setPercentage(66);
            fRemove.setInputFormat(rData);
            weka.core.Instances testHoldout = Filter.useFilter(rData, fRemove);

            fRemove.setPercentage(66);
            fRemove.setInvertSelection(true);
            fRemove.setInputFormat(rData);
            weka.core.Instances trainHoldout = Filter.useFilter(rData, fRemove);

            evalHO.evaluateModel(model, testHoldout);
        }

        pw.println("-------------------- HOLDOUT --------------------");
        pw.println(evalHO.toSummaryString());
        pw.println(evalHO.toClassDetailsString());
        pw.println(evalHO.toMatrixString());
        pw.println("");


        // cross-validation 5-fold
        Evaluation evalKFCV = new Evaluation(dataMini);
        evalKFCV.crossValidateModel(model, dataMini, 5, new Random(1));

        pw.println("-------------------- CROSS-VALIDATION --------------------");
        pw.println(evalKFCV.toSummaryString());
        pw.println(evalKFCV.toClassDetailsString());
        pw.println(evalKFCV.toMatrixString());
        pw.close();
    }

    private static Instances datuakTxikitu(Instances data, double p) throws Exception{
        //System.out.println("Datu kopurua txikitu baino lehen " + data.numInstances());
        Randomize fRandom = new Randomize();
        fRandom.setRandomSeed(1);
        fRandom.setInputFormat(data);
        weka.core.Instances rData = Filter.useFilter(data, fRandom);

        RemovePercentage fRemove = new RemovePercentage();
        fRemove.setPercentage(p*100);
        fRemove.setInputFormat(rData);
        weka.core.Instances datuTxiki = Filter.useFilter(rData, fRemove);

        //System.out.println("Datuak txikitu ondoren: " + datuTxiki.numInstances());

        return datuTxiki;
    }
}