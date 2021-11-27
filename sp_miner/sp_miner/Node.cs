using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace sp_miner
{
    class Node
    {
        public int id { get; set; } // node id
        public List<string> itemset { get; set; } // itemset contained in the node
        public Dictionary<int, List<int>>[] tidposset { get; set; } // trans_id: list of positions
        public int sup { get; set; } // support of the node

        public Node(int n_labels)
        {
            this.id = -1;
            this.itemset = new List<string>();
            this.tidposset = new Dictionary<int, List<int>>[n_labels];
            for (int x = 0; x < n_labels; x++)
            {
                this.tidposset[x] = new Dictionary<int, List<int>>();
            }
            this.sup = 0;
        }
    }
}
