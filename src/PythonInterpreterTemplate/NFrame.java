/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/AWTForms/Frame.java to edit this template
 */
package PythonInterpreterTemplate;
import java.awt.Button;
import java.awt.Checkbox;
    import javax.swing.*;    
    import java.awt.event.*;    
    import java.io.*;    
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.border.EtchedBorder;
import javax.swing.border.TitledBorder;
    
    public class NFrame extends JFrame implements ActionListener{
    public static String filepath,file_labels_path,file_save_path,file_name;
    
    public static boolean cccp=true;
    ButtonGroup group = new ButtonGroup();
    JMenuBar mb;    
    JMenu file;   
    JMenu file_save;   
    JMenuItem open; 
    JMenuItem save;
    JMenuItem n_labels;
    public JTextArea ta; 
    Button ta4;
    JRadioButton tal;
    JRadioButton ta1;
    JRadioButton ta2;
    JRadioButton ta3;
    JRadioButton ta5;
    
    NFrame(){
    
    open=new JMenuItem("Open File");    
    open.addActionListener(this);
    save=new JMenuItem("Save File Not For Doc2vecC");    
    save.addActionListener(this);
    n_labels=new JMenuItem("Labels For Doc2vecC");    
    n_labels.addActionListener(this);
    file=new JMenu("File");    
    file.add(open); 
    file.add(save);
    file.add(n_labels);
    mb=new JMenuBar();    
    mb.setBounds(0,0,800,20);    
    mb.add(file);                  
    ta=new JTextArea( 37, 58 );
    ta.setEditable ( false );
    JScrollPane scroll = new JScrollPane(ta);
    JPanel middlePanel = new JPanel ();
    middlePanel.setBorder ( new TitledBorder ( new EtchedBorder (), "Display Area" ) );
    scroll.setVerticalScrollBarPolicy(ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);
    middlePanel.add ( scroll );
    middlePanel.setBounds(0,20,700,700);
    
     tal = new JRadioButton("Sep Cluster");
     ta1 = new JRadioButton("Sep Classify");
     ta2 = new JRadioButton("Sim Classify");
     ta3 = new JRadioButton("Sim Cluster");
     ta5 = new JRadioButton("Doc2VecC Classify");
    ta4=new Button("lets go !");
    ta4.addActionListener(this);
    ta4.setBounds(300,500,100,20);
    tal.setBounds(100,0,100,20);
    ta1.setBounds(200,0,100,20);
    ta2.setBounds(300,0,100,20);
    ta3.setBounds(400,0,100,20);
    ta4.setBounds(600,0,100,20);
    ta5.setBounds(500,0,100,20);
    ta1.addActionListener(this);
    group.add(tal);
    group.add(ta1);
    group.add(ta2);
    group.add(ta3);
    group.add(ta5);
    
    add(mb);
    add(tal);    
    add(ta1);
    add(ta2);
    add(ta3);
    add(ta5);
    add(ta4);
    add(mb);    
    
    add(middlePanel);
    
    }
    @Override
    public void actionPerformed(ActionEvent e) {
        
    if((e.getSource()==ta4) && ((ta1.isSelected()) || (ta2.isSelected())|| (tal.isSelected()) || (ta3.isSelected()))){
        try{
        PythonInterpreterTemplate.add_prefix_for_god_sake(filepath, file_save_path, 2, ta);
         
        }catch(java.lang.NullPointerException fe){
           JOptionPane.showMessageDialog(new JFrame(),"Input your Sequences file first File->open File->ok","Error Message Box",JOptionPane.ERROR_MESSAGE); 
        } catch (Throwable ex) {
            Logger.getLogger(NFrame.class.getName()).log(Level.SEVERE, null, ex);
        }
        if(tal.isSelected()){
            try {
                PythonInterpreterTemplate.HUS2Vec_sep_cluster(file_name,ta);
            } catch (InterruptedException ex) {
                Logger.getLogger(NFrame.class.getName()).log(Level.SEVERE, null, ex);
            } catch (Throwable ex) {
                Logger.getLogger(NFrame.class.getName()).log(Level.SEVERE, null, ex);
            }
        }else if(ta1.isSelected()){
            try {
                PythonInterpreterTemplate.HUS2Vec_sep_classify(file_name,ta);
            } catch (InterruptedException ex) {
                Logger.getLogger(NFrame.class.getName()).log(Level.SEVERE, null, ex);
            } catch (Throwable ex) {
                Logger.getLogger(NFrame.class.getName()).log(Level.SEVERE, null, ex);
            }
        }else if(ta2.isSelected()){
            try {
                PythonInterpreterTemplate.HUS2Vec_sim_classify(file_name,ta);
             } catch (InterruptedException ex) {
                Logger.getLogger(NFrame.class.getName()).log(Level.SEVERE, null, ex);
            } catch (Throwable ex) {
                Logger.getLogger(NFrame.class.getName()).log(Level.SEVERE, null, ex);
            }
        }else if(ta3.isSelected()){
            try {
                PythonInterpreterTemplate.HUS2Vec_sim_cluster(file_name,ta);
            } catch (InterruptedException ex) {
                Logger.getLogger(NFrame.class.getName()).log(Level.SEVERE, null, ex);
            } catch (Throwable ex) {
                Logger.getLogger(NFrame.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
         
    }if(ta5.isSelected() &&(e.getSource()==ta4)){
            try {
                PythonInterpreterTemplate.train_doc2vecC_classify(ta,file_labels_path,filepath);
            } catch (InterruptedException ex) {
                Logger.getLogger(NFrame.class.getName()).log(Level.SEVERE, null, ex);
            } catch (Throwable ex) {
                Logger.getLogger(NFrame.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    
    if(e.getSource()==open){    
        JFileChooser fc=new JFileChooser();    
        int i=fc.showOpenDialog(this);    
        if(i==JFileChooser.APPROVE_OPTION){    
            File f=fc.getSelectedFile();    
            filepath=f.getPath().replace("\\","\\\\");;    
            }
    }
        if(e.getSource()==save){    
        JFileChooser fcc=new JFileChooser();    
        int x=fcc.showOpenDialog(this);    
        if(x==JFileChooser.APPROVE_OPTION){    
            File fe=fcc.getSelectedFile();
            file_save_path=fe.getPath().replace("\\","\\\\");
            file_name=fe.getName().replace(".txt", "");
            
            
        }    
    }
        if(e.getSource()==n_labels){    
        JFileChooser fcc1=new JFileChooser();    
        int x=fcc1.showOpenDialog(this);    
        if(x==JFileChooser.APPROVE_OPTION){    
            File fe=fcc1.getSelectedFile();
            file_labels_path=fe.getPath().replace("\\","\\\\");
            
            
            
        }    
    }
    }
    public static void main(String[] args) {    
        NFrame om=new NFrame();    
                 om.setSize(700,700);    
                 om.setLayout(null);    
                 om.setVisible(true);    
                 om.setDefaultCloseOperation(EXIT_ON_CLOSE);    
    
    }    
    }  
