package zati1;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;


public class BateragarriEgin {
    public static void main(String[] args) throws Exception{
        // args[0]: train filtratuaren path-a
        // args[1]: test filtratuaren path-a
        // args[2]: dev fitxategi filtratu (train.BOW.FSS) eta bateragarria gordetzeko path-a

        /*
        String args0 = "/home/mikel/Documentos/DATATA/arff/datuFiltratuak/train.arff";
        String args1 = "/home/mikel/Documentos/DATATA/arff/datuFiltratuak/test.arff";
        String args2 = "/home/mikel/Documentos/DATATA/arff/datuFiltratuak/testBat.arff";
        */

        DataSource srcTrain = new DataSource(args[0]);
        DataSource srcDev = new DataSource(args[1]);
        Instances train = srcTrain.getDataSet();
        Instances dev = srcDev.getDataSet();

        System.out.println("Test numAtt bateragarria izan gabe: " + dev.numAttributes());
        Remove remove = new Remove();
        remove.setInputFormat(train);
        Instances devBat = Filter.useFilter(dev, remove);

        System.out.println("dev eta train bateragarriak dira: " + train.equalHeaders(devBat));

        System.out.println("train numAtt: " + train.numAttributes() + "   test NumAtt: " + devBat.numAttributes());
        System.out.println("train numIns: " + train.numInstances() + "   test numIns: " + devBat.numInstances());
        arffSortu(args[2], devBat);
    }

    private static void arffSortu(String path, Instances datuak) throws Exception{
        ArffSaver saver = new ArffSaver();
        saver.setInstances(datuak);
        saver.setFile(new File(path));
        saver.writeBatch();
    }
}