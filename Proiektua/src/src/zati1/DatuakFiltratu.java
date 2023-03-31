package zati1;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;

public class DatuakFiltratu {
    public static void main(String[] args) throws Exception {
        // args[0]: filtratu nahi diren datuen path-a
        // args[1]: datuak filtratuta gordetzeko path-a

/*
        String args0 = "/home/mikel/Documentos/DATATA/arff/datuGordinak/test.arff";
        String args1 = "/home/mikel/Documentos/DATATA/arff/datuFiltratuak/test.arff";
        */

        DataSource source = new DataSource(args[0]);
        Instances data = source.getDataSet();


        StringToWordVector filtroSTWV = new StringToWordVector();

        String bannedChars = "\\/'%$&#,.-:;?!()";
        WordTokenizer wt = new WordTokenizer();
        wt.setDelimiters(bannedChars);

        filtroSTWV.setLowerCaseTokens(true);
        filtroSTWV.setOutputWordCounts(true);
        filtroSTWV.setWordsToKeep(5000);
        filtroSTWV.setTokenizer(wt);
        filtroSTWV.setInputFormat(data);
        Instances data_BOW = Filter.useFilter(data, filtroSTWV);
        data_BOW.setClassIndex(0);

        System.out.println("STWV ondoren atributu kopurua: " + data_BOW.numAttributes());

        AttributeSelection filtroAS = new AttributeSelection();
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        Ranker search = new Ranker();
        search.setNumToSelect(1000);
        filtroAS.setEvaluator(eval);
        filtroAS.setSearch(search);
        filtroAS.setInputFormat(data_BOW);

        Instances dataFiltratuta = Filter.useFilter(data_BOW, filtroAS);

        Remove rm = new Remove();
        int[] indizeak = {0,1,2,3,4,5,6,7,8,9};
        rm.setAttributeIndicesArray(indizeak);
        rm.setInputFormat(dataFiltratuta);

        Instances dataEmaitza = Filter.useFilter(dataFiltratuta, rm);

        System.out.println("AS ondoren atributu kopurua: " + dataFiltratuta.numAttributes());

        arffSortu(args[1], dataEmaitza);
    }

    private static void arffSortu(String path, Instances datuak) throws Exception{
        ArffSaver saver = new ArffSaver();
        saver.setInstances(datuak);
        saver.setFile(new File(path));
        saver.writeBatch();
    }
}