package zati1;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.File;

public class DatuakZatitu {

    public static void main(String[] args) throws Exception {
        // args[0]: datu guztiak dauden arff fitxategiaren path-a
        // args[1]: train gordetzeko path-a
        // args[2]: dev gordetzeko path-a
        // args[3]: test gordetzeko path-a

        // String args0 = "/home/mikel/Documentos/DATATA/arff/mail_spam.arff";
        // String args1 = "/home/mikel/Documentos/DATATA/arff/datuGordinak/train.arff";
        // String args2 = "/home/mikel/Documentos/DATATA/arff/datuGordinak/dev.arff";
        // String args3 = "/home/mikel/Documentos/DATATA/arff/datuGordinak/test.arff";

        DataSource source = new DataSource(args[0]);
        Instances data = source.getDataSet();

        Randomize randomFilter = new Randomize();
        randomFilter.setRandomSeed(1);
        randomFilter.setInputFormat(data);
        Instances dataR = Filter.useFilter(data, randomFilter);

        // Use the RemovePercentage filter to split the data
        RemovePercentage trainFilter = new RemovePercentage();
        trainFilter.setInputFormat(dataR);
        trainFilter.setPercentage(30);
        Instances train = Filter.useFilter(dataR, trainFilter);

        RemovePercentage devFilter = new RemovePercentage();
        devFilter.setInputFormat(dataR);
        devFilter.setPercentage(30);
        devFilter.setInvertSelection(true);
        Instances development = Filter.useFilter(dataR, devFilter);

        RemovePercentage testFilter = new RemovePercentage();
        testFilter.setInputFormat(development);
        testFilter.setPercentage(50);
        Instances test = Filter.useFilter(development, testFilter);

        testFilter.setInputFormat(development);
        testFilter.setPercentage(50);
        testFilter.setInvertSelection(true);
        Instances dev = Filter.useFilter(development, testFilter);

        System.out.println("Train instantzia kopurua: " + train.numInstances());
        System.out.println("Test instantzia kopurua: " + test.numInstances());
        System.out.println("Development instantzia kopurua: " + dev.numInstances());

        arffSortu(args[1], train);
        arffSortu(args[2], dev);
        arffSortu(args[3], test);

    }

    private static void arffSortu(String path, Instances datuak) throws Exception{
        ArffSaver saver = new ArffSaver();
        saver.setInstances(datuak);
        saver.setFile(new File(path));
        saver.writeBatch();
    }
}