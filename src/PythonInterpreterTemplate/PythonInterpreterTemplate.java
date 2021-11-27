/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package PythonInterpreterTemplate;
//import static PythonInterpreterTemplate.NFrame.filepath;
import org.python.core.*;
import org.python.util.PythonInterpreter;  
import java.awt.*;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.StringWriter;
import java.util.concurrent.TimeUnit;
import javax.swing.JTextArea;
import jep.*;
import jep.python.*;
import jep.Jep;
import jep.JepConfig;
import jep.JepException;

import jep.SharedInterpreter;
import static org.python.util.jython.logger;


/**
 *
 * @author admin
 */

public class PythonInterpreterTemplate {
        

    public static String train_doc2vecC_classify( JTextArea tobeappended,String n_labels_path,String path_dataset_raw) throws Throwable  
   {  
       
       jep.JepConfig jepConf = new  JepConfig();
       
           jepConf.addIncludePaths(System.getProperty("user.dir")+"\\HUS2Vec");
           ByteArrayOutputStream out = new ByteArrayOutputStream();
           jepConf.redirectStdout(out);
           

    try (Interpreter interp = jepConf.createSubInterpreter()) {

        interp.eval("import train_doc2vecC_classify as p");
        interp.eval("p.Doc2VecC_run(\""+path_dataset_raw+"\",\""+n_labels_path+"\",100)");
          
        
        tobeappended.append(out.toString());
    } catch (JepException t) {
        System.out.print(t);
    }
      return out.toString();
   }
    public static String add_prefix_for_god_sake( String inp,String outp,Integer lnb,JTextArea tobeappended ) throws Throwable  
   {  
       
       jep.JepConfig jepConf = new  JepConfig();
       
           jepConf.addIncludePaths(System.getProperty("user.dir")+"\\HUS2Vec");
           ByteArrayOutputStream out = new ByteArrayOutputStream();
           jepConf.redirectStdout(out);
           

    try (Interpreter interp = jepConf.createSubInterpreter()) {

        interp.eval("import add_prefix_for_god_sake as p");
        interp.eval("p.add_prefix().run('"+inp+"','"+outp+"',"+lnb+")");
          
        
        tobeappended.append(out.toString());
    } catch (JepException t) {
        System.out.print(t);
    }
    
      return out.toString();
   }
    public static String HUS2Vec_sim_cluster(String spath,JTextArea tobeappended) throws Throwable  
   {  
       
       jep.JepConfig jepConf = new  JepConfig();
       
           jepConf.addIncludePaths(System.getProperty("user.dir")+"\\HUS2Vec");
           ByteArrayOutputStream out = new ByteArrayOutputStream();
           jepConf.redirectStdout(out);
           

    try (Interpreter interp = jepConf.createSubInterpreter()) {

        interp.eval("import HUS2Vec_sim_cluster as p");
        interp.eval("p.data_name='"+spath+"'");
        interp.eval("p.sim_training_c().run()");
          
        
        tobeappended.append(out.toString());
    } catch (JepException t) {
        System.out.print(t);
    }
      return out.toString();
   }
    public static String HUS2Vec_sep_cluster(String spath,JTextArea tobeappended) throws Throwable  
   {  
       
       jep.JepConfig jepConf = new  JepConfig();
       
           jepConf.addIncludePaths(System.getProperty("user.dir")+"\\HUS2Vec");
           ByteArrayOutputStream out = new ByteArrayOutputStream();
           jepConf.redirectStdout(out);
           

    try (Interpreter interp = jepConf.createSubInterpreter()) {

        interp.eval("import HUS2Vec_sep_cluster as p");
        interp.eval("p.data_name='"+spath+"'");
        interp.eval("p.sep_training_c().run()");
          
        
        tobeappended.append(out.toString());
    } catch (JepException t) {
        System.out.print(t);
    }
      return out.toString();
   }
    public static String HUS2Vec_sep_classify(String spath,JTextArea tobeappended) throws Throwable  
   {  
       
       jep.JepConfig jepConf = new  JepConfig();
       
           jepConf.addIncludePaths(System.getProperty("user.dir")+"\\HUS2Vec");
           ByteArrayOutputStream out = new ByteArrayOutputStream();
           jepConf.redirectStdout(out);
           

    try (Interpreter interp = jepConf.createSubInterpreter()) {

        interp.eval("import HUS2Vec_sep_classify as p");
        interp.eval("p.data_name='"+spath+"'");
        interp.eval("p.sep_training().run()");
          
        
        tobeappended.append(out.toString());
    } catch (JepException t) {
        System.out.print(t);
    }
      return out.toString();
   }
    public static String HUS2Vec_sim_classify(String spath,JTextArea tobeappended) throws Throwable  
   {  
       System.out.println(spath);
       jep.JepConfig jepConf = new  JepConfig();
       
           jepConf.addIncludePaths(System.getProperty("user.dir")+"\\HUS2Vec");
           ByteArrayOutputStream out = new ByteArrayOutputStream();
           jepConf.redirectStdout(out);
           

    try (Interpreter interp = jepConf.createSubInterpreter()) {

        interp.eval("import HUS2Vec_sim_classify as p");
        
        interp.eval("p.data_name='"+spath+"'");
        System.out.println(interp.getValue("p.path"));
        interp.eval("p.sim_training().run()");
          
        
        tobeappended.append(out.toString());
    } catch (JepException t) {
        System.out.print(t);
    }
      return out.toString();
   }
   public static void main( String gargs[] ) throws Throwable  
   {  
       
        
        NFrame.main(gargs);
      
   }
   
} 



