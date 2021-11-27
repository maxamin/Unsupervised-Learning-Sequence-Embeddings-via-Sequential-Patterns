using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace sp_miner
{
    class Dataset
    {
        public int n_row { get; set; }
        public int n_label { get; set; }
        public List<string> items { get; set; }
        public double avg_len { get; set; } // average transaction length
        public Dictionary<int, int> rows_labels { get; set; } // {<label id>: # of transactions containing label id}
        public List<List<string>> data { get; set; }
        public List<int> labels { get; set; } // store label id of each transaction
        public Dictionary<string, int> dict_label { get; set; } // map a label (string) to a label id (int)

        // the first column contains labels
        public Dataset()
        {
            this.n_row = 0;
            this.n_label = 0;
            this.items = new List<string>();
            this.avg_len = 0;
            this.rows_labels = new Dictionary<int, int>();
            this.data = new List<List<string>>();
            this.labels = new List<int>();
            this.dict_label = new Dictionary<string, int>();
        }

        public void loadData(string file, char sep)
        {
            List<string> s_labels = new List<string>();
            using (StreamReader sr = File.OpenText(file))
            {
                string line = "";
                while ((line = sr.ReadLine()) != null)
                {
                    line = line.Trim();
                    string[] content = line.Split('\t');
                    if (content.Count() > 1)
                    {
                        s_labels.Add(content[0]);
                        addTransaction(content[1].Split(sep));
                    }
                }
            }
            this.n_row = this.data.Count;
            // compute average transaction length
            this.avg_len = Math.Round((double)this.avg_len / this.n_row, 2);
            // convert labels from string to int
            List<string> distinct_labels = s_labels.Distinct().ToList();
            this.dict_label = distinct_labels.Select((s, i) => new { s, i }).ToDictionary(x => x.s, x => x.i);
            foreach (string label in s_labels)
            {
                this.labels.Add(dict_label[label]);
            }
            this.n_label = distinct_labels.Count;
            this.rows_labels = this.labels.GroupBy(x => x).ToDictionary(g => g.Key, g => g.Count());
        }

        void addTransaction(string[] s_items)
        {
            int len = s_items.Length;
            List<string> itemset = new List<string>();
            // start from element 0 because label was handed above
            for (int i = 0; i < len; i++)
            {
                string item = s_items[i];
                // don't ignore duplicated items in a transaction
                itemset.Add(item);
                // obtain unique items in the whole dataset    
                if (!this.items.Contains(item))
                {
                    this.items.Add(item);
                }
            }
            this.avg_len += itemset.Count;
            this.data.Add(itemset);
        }
    }
}
