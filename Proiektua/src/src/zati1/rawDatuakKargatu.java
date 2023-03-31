package zati1;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.TextDirectoryLoader;

import java.io.File;

public class rawDatuakKargatu {
    public static void main(String[] args) throws Exception{
        // args[0] txt formatuean dauden datuen path-a
        // args[1] datuak non gorde nahi diren arff fitxategiaren path-a

        //String arg0 = "/home/mikel/Documentos/DATATA/mail_spam";
        //String arg1 = "/home/mikel/Documentos/DATATA/arff/mail_spam.arff";

        Instances data = null;
        TextDirectoryLoader loader = new TextDirectoryLoader();
        loader.setSource(new File(args[0]));
        data = loader.getDataSet();
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(args[1]));
        saver.writeBatch();
    }
}
