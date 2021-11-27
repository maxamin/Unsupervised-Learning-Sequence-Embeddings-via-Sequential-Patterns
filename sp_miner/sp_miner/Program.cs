using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.IO;

namespace sp_miner
{
    class Program
    {
        // # nodes on tree
        static int nNode = 1;
        // max length of sequential patterns
        static int maxLen = int.MaxValue;
        // list of SPs
        static List<Node> SPs = new List<Node>();

        #region parse parameters
        private static int ArgPos(string str, string[] args)
        {
            for (int a = 0; a < args.Length; a++)
            {
                if (str.Equals(args[a]))
                {
                    if (a == args.Length - 1)
                    {
                        throw new ArgumentException(string.Format("Argument missing for {0}", str));
                    }

                    return a;
                }
            }

            return -1;
        }
        #endregion

        #region intersect of two tidsets
        static List<int> intersectTidsets(List<int> a1, List<int> a2)
        {
            List<int> a = new List<int>();
            if (a1.Count == 0 || a2.Count == 0)
            {
                return a;
            }

            int i = 0, j = 0;
            a2.Add(a1[a1.Count - 1] + 1);
            while (i < a1.Count)
            {
                if (a1[i] < a2[j])
                {
                    i++;
                }
                else if (a1[i] > a2[j])
                {
                    j++;
                }
                else
                {
                    a.Add(a1[i++]);
                    j++;
                }
            }
            a2.RemoveAt(a2.Count - 1);

            return a;
        }
        #endregion

        #region intersect of two tidpossets
        static Dictionary<int, List<int>> intersectTidpossets(Dictionary<int, List<int>> b1, Dictionary<int, List<int>> b2, List<int> a, int gap)
        {
            Dictionary<int, List<int>> b = new Dictionary<int, List<int>>();
            foreach (int tid in a)
            {
                List<int> b1_positions = b1[tid];
                List<int> b2_positions = b2[tid];
                List<int> b_positions = new List<int>();
                if (gap > 0) // check the gap between items
                {
                    foreach (int b2_pos in b2_positions)
                    {
                        foreach (int b1_pos in b1_positions)
                        {
                            if (b2_pos > b1_pos & b2_pos <= b1_pos + gap)
                            {
                                b_positions.Add(b2_pos);
                                break;
                            }
                        }
                    }
                }
                else // don't consider the gap between items
                {
                    foreach (int b2_pos in b2_positions)
                    {
                        foreach (int b1_pos in b1_positions)
                        {
                            if (b2_pos > b1_pos)
                            {
                                b_positions.Add(b2_pos);
                                break;
                            }
                        }
                    }
                }

                if (b_positions.Count > 0)
                {
                    b.Add(tid, b_positions);
                }
            }

            return b;
        }
        #endregion

        #region union of two itemsets
        static List<string> unionItemsets(List<string> c1, List<string> c2)
        {
            List<string> c = new List<string>(c1);
            c.Add(c2[c2.Count - 1]);

            return c;
        }
        #endregion

        #region find singleton candidates and their tidposset
        static Dictionary<string, Node> findSingletonCandidates(Dataset dt_data)
        {
            Dictionary<string, Node> SC = new Dictionary<string, Node>();
            for (int tid = 0; tid < dt_data.n_row; tid++)
            {
                int label = dt_data.labels[tid];
                // there are duplicated items in a transaction
                for (int i = 0; i < dt_data.data[tid].Count; i++)
                {
                    string it = dt_data.data[tid][i]; // an item
                    if (!SC.Keys.Contains(it)) // new item
                    {
                        Node node = new Node(dt_data.n_label);
                        node.itemset.Add(it);
                        node.tidposset[label].Add(tid, new List<int>());
                        node.tidposset[label][tid].Add(i);
                        SC.Add(it, node);
                    }
                    else // old item
                    {
                        // this item is in a different transaction
                        if (!SC[it].tidposset[label].Keys.Contains(tid))
                        {
                            SC[it].tidposset[label].Add(tid, new List<int>());
                            SC[it].tidposset[label][tid].Add(i);
                        }
                        else // this item is in the same transaction
                        {
                            SC[it].tidposset[label][tid].Add(i);
                        }
                    }
                }
            }

            return SC;
        }
        #endregion                      

        #region find sequential 1-patterns
        static List<Node> findSequential1Items(List<Node> SC, int n_label, double minSup)
        {
            List<Node> SPs = new List<Node>();
            foreach (Node node in SC)
            {
                for (int label = 0; label < n_label; label++)
                {
                    // compute support   
                    node.sup += node.tidposset[label].Count;
                }
                // check if this node is frequent
                if (node.sup >= minSup)
                {
                    node.id = nNode;
                    // increase # nodes on tree
                    nNode++;
                    SPs.Add(node);
                }
            }

            return SPs;
        }
        #endregion

        #region find sequential k-patterns
        static void findSequentialkItems(List<Node> Lr, double minSup, int gap, int n_label, Dictionary<int, int> rows_labels, int level)
        {
            if (level > maxLen)
            {
                return;
            }
            foreach (Node li in Lr)
            {
                List<Node> P_i = new List<Node>();
                foreach (Node lj in Lr)
                {
                    Node O = new Node(n_label);
                    // compute tidposset
                    for (int label = 0; label < n_label; label++)
                    {
                        List<int> O_tidset = intersectTidsets(li.tidposset[label].Keys.ToList(), lj.tidposset[label].Keys.ToList());
                        O.tidposset[label] = intersectTidpossets(li.tidposset[label], lj.tidposset[label], O_tidset, gap);
                        // compute support
                        O.sup += O.tidposset[label].Count;
                    }
                    // check if this node is frequent
                    if (O.sup >= minSup)
                    {
                        O.id = nNode;
                        nNode++;
                        O.itemset = unionItemsets(li.itemset, lj.itemset);
                        P_i.Add(O);
                        SPs.Add(O);
                    }
                }
                if (P_i.Count > 0)
                {
                    findSequentialkItems(P_i, minSup, gap, n_label, rows_labels, ++level);
                }
            }
        }
        #endregion        

        #region write sequential patterns to file
        static void writeSPs(string file_itemset, List<Node> SPs, int n_row)
        {
            using (StreamWriter sw = new StreamWriter(file_itemset))
            {
                // write header
                sw.WriteLine("id,itemset,size,sup");
                // write itemsets and properties
                foreach (Node node in SPs)
                {
                    string itemset = string.Join(" ", node.itemset);
                    int size = node.itemset.Count;
                    double sup = Math.Round((double)node.sup / n_row, 4);
                    sw.WriteLine(node.id + "," + itemset + "," + size + "," + sup);
                }
            }
        }
        #endregion        

        #region write sequences covered by sequential patterns to file
        static void writeTrainSPs(string file_train, List<Node> SPs, Dataset dt_train)
        {
            // map a transaction to sequential patterns
            using (StreamWriter sw = new StreamWriter(file_train))
            {
                for (int i = 0; i < dt_train.n_row; i++)
                {
                    int label = dt_train.labels[i];
                    List<string> itemset = new List<string>();
                    foreach (Node node in SPs)
                    {
                        if (node.tidposset[label].Keys.Contains(i))
                        {
                            itemset.Add(node.id.ToString());
                        }
                    }
                    // convert a label from int to string
                    string s_label = dt_train.dict_label.FirstOrDefault(x => x.Value == label).Key;
                    sw.WriteLine(s_label + "\t" + string.Join(" ", itemset));
                }
            }
        }
        #endregion        

        #region write sequences covered by symbols and SPs to file
        static void writeTrainItemsSPs(string file_train, List<Node> SPs, Dataset dt_train)
        {
            // map a transaction to items and SPs
            using (StreamWriter sw = new StreamWriter(file_train))
            {
                for (int i = 0; i < dt_train.n_row; i++)
                {
                    int label = dt_train.labels[i];
                    List<string> itemset = new List<string>();
                    itemset.AddRange(dt_train.data[i]);
                    foreach (Node node in SPs)
                    {
                        if (node.tidposset[label].Keys.Contains(i))
                        {
                            itemset.Add(node.id.ToString());
                        }
                    }
                    // convert a label from int to string
                    string s_label = dt_train.dict_label.FirstOrDefault(x => x.Value == label).Key;
                    sw.WriteLine(s_label + "\t" + string.Join(" ", itemset));
                }
            }
        }
        #endregion                

        static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                Console.WriteLine("mining sequential patterns (SPs)...");
                Console.WriteLine("\t-dataset <file>");
                Console.WriteLine("\tuse sequences from <file> to mine SPs");
                Console.WriteLine("\t-minsup <float>");
                Console.WriteLine("\tset minimum support threshold in [0,1]; default is 0.5");
                Console.WriteLine("\t-gap <int>");
                Console.WriteLine("\tset gap constraint > 0; set 0 if don't use gap constraint");
                Console.WriteLine("\t-sp <file>");
                Console.WriteLine("\tsave discovered SPs to <file> (optional)");
                Console.WriteLine("\t-seqsp <file>");
                Console.WriteLine("\tconvert each sequence to a set of SPs and save it to <file> (optional)");
                Console.WriteLine("\t-seqsymsp <file>");
                Console.WriteLine("\tconvert each sequence to a set of symbols and SPs and save it to <file> (optional)");

                return;
            }
            int para_id = 0;
            string in_seq = "";
            double r_minSup = 0;
            int gap = 0;
            string out_sp = "";
            string out_seq_sp = "";
            string out_seq_sym_sp = "";
            if ((para_id = ArgPos("-dataset", args)) > -1)
            {
                in_seq = args[para_id + 1];
            }
            if ((para_id = ArgPos("-minsup", args)) > -1)
            {
                r_minSup = 0.03;
            }
            if ((para_id = ArgPos("-gap", args)) > -1)
            {
                gap = int.Parse(args[para_id + 1]);
            }
            if ((para_id = ArgPos("-sp", args)) > -1)
            {
                out_sp = args[para_id + 1];
            }
            if ((para_id = ArgPos("-seqsp", args)) > -1)
            {
                out_seq_sp = args[para_id + 1];
            }
            if ((para_id = ArgPos("-seqsymsp", args)) > -1)
            {
                out_seq_sym_sp = args[para_id + 1];
            }

            // load sequence dataset
            Dataset dt_data = new Dataset();
            dt_data.loadData(in_seq, ' ');
            double[] label_dist = new double[dt_data.n_label];
            for (int label = 0; label < dt_data.n_label; label++)
            {
                label_dist[label] = (double)dt_data.rows_labels[label] / dt_data.n_row;
                label_dist[label] = Math.Round(label_dist[label] * 100, 2);
            }
            Console.WriteLine(in_seq + ": #sequences=" + dt_data.n_row + ", #symbols=" + dt_data.items.Count +
                ", avg length=" + dt_data.avg_len + ", #labels=" + dt_data.n_label + ", label dist.=" + string.Join("&", label_dist));

            Stopwatch sw = Stopwatch.StartNew();
            // reset values
            nNode = 1;
            SPs = new List<Node>();
            // compute absolute minSup
            double minSup = r_minSup * dt_data.n_row;
            // find singleton candidates and their tidposset
            Dictionary<string, Node> SC = findSingletonCandidates(dt_data);
            // find sequential 1-patterns
            SPs = findSequential1Items(SC.Values.ToList<Node>(), dt_data.n_label, minSup);
            // create root node
            List<Node> Lr = new List<Node>(SPs);
            // find sequential k-patterns
            findSequentialkItems(Lr, minSup, gap, dt_data.n_label, dt_data.rows_labels, 2);
            Console.WriteLine("minSup: " + r_minSup + ", SPs: " + SPs.Count);
            // write sequential patterns to file
            if (out_sp != "")
            {
                if (SPs.Count > 0)
                {
                    writeSPs(out_sp, SPs, dt_data.n_row);
                }
            }
            // write sequences covered by sequential patterns to file
            if (out_seq_sp != "")
            {
                if (SPs.Count > 0)
                {
                    writeTrainSPs(out_seq_sp, SPs, dt_data);
                }
            }
            // write sequences covered by symbols and SPs to file
            if (out_seq_sym_sp != "")
            {
                if (SPs.Count > 0)
                {
                    writeTrainItemsSPs(out_seq_sym_sp, SPs, dt_data);
                }
            }
            sw.Stop();
            long time = sw.ElapsedMilliseconds;
            Console.WriteLine("runtime: " + time / 1000.0 + " (s)");
        }
    }
}
