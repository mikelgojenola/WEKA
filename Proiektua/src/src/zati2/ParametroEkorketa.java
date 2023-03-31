package zati2;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

public class ParametroEkorketa {
    public static void main(String[] args) throws Exception {

        // args[0]: train datu filtratuen fitxategiaren path-a
        // args[1]: learning rate parametroaren balio optimoa gordetzeko txt fitxategiaren path-a
        // args[0]: hiddenLayers parametroaren balio optimoa gordetzeko txt fitxategiaren path-a


        DataSource source = new DataSource(args[0]);
        Instances train = source.getDataSet();
        train.setClassIndex(train.numAttributes()-1);

        int minId = Utils.minIndex(train.attributeStats(train.classIndex()).nominalCounts);
        System.out.println("Klase minoritarioa: " + train.attribute(train.classIndex()).value(minId));

        ArrayList<String> hLlista = new ArrayList<String>();
        hLlista.add("30,30,30");
        hLlista.add("70");
        hLlista.add("100,40");
        hLlista.add("200,120,50");

        double kalitateOpt = 0.0;
        String hiddenLayerOpt = null;
        double lROpt = 0.0;

        int iterazioKop = 1;

        for(double lr = 0.05; lr < 0.5; lr += 0.1){
            for (String hL : hLlista){

                System.out.println("loop: " + iterazioKop);

                System.out.println("Learning Rate: " + lr);
                System.out.println("Hidden Layers: " + hL);

                MultilayerPerceptron cls = new MultilayerPerceptron();

                cls.setHiddenLayers(hL);
                cls.setLearningRate(lr);
                cls.setTrainingTime(150);

                Randomize filterRandomize = new Randomize();
                filterRandomize.setRandomSeed(iterazioKop);
                filterRandomize.setInputFormat(train);
                train = Filter.useFilter(train, filterRandomize);

                RemovePercentage rmpct = new RemovePercentage();
                rmpct.setInvertSelection(false);
                rmpct.setPercentage(33);
                rmpct.setInputFormat(train);
                Instances trainF = Filter.useFilter(train, rmpct);

                Instances test;
                RemovePercentage rmpct2 = new RemovePercentage();
                rmpct2.setInvertSelection(true);
                rmpct2.setPercentage(33);
                rmpct2.setInputFormat(train);
                test = Filter.useFilter(train, rmpct2);

                cls.buildClassifier(trainF);

                Evaluation eval = new Evaluation(trainF);
                eval.evaluateModel(cls, test);

                double fMeasure = eval.fMeasure(minId);
                System.out.println("F-measure klase minoritarioarekiko: " + fMeasure);

                if (fMeasure > kalitateOpt){
                    kalitateOpt = fMeasure;
                    hiddenLayerOpt = hL;
                    lROpt = lr;
                }

                iterazioKop++;
            }
        }
        parametroakGorde(lROpt, hiddenLayerOpt, args[1], args[2]);

    }

    private static void parametroakGorde(double lR, String hL, String lRpath, String hLpath) throws Exception{
        FileWriter fw = new FileWriter(new File(lRpath));
        PrintWriter pw = new PrintWriter(fw);
        pw.print(lR);
        pw.close();

        FileWriter fw2 = new FileWriter(new File(hLpath));
        PrintWriter pw2 = new PrintWriter(fw2);
        pw2.print(hL);
        pw2.close();
    }
}
