import os
from optparse import OptionParser
import sys
parser = OptionParser()
usage = "usage: python "+ __file__+" -n 2 --output outputfile.txt --input inputfile.txt"

parser.add_option("-o", "--output", type="string",
                  help="select output file name",
                  dest="out")

parser.add_option("-i", "--input", type="string",
                  help="select input file name",
                  dest="inp")
parser.add_option("-n", "--nlabels", type="int",
                  help="number of labels",
                  dest="nlb")

(options, args) = parser.parse_args()
class add_prefix:
    def __call__(self):
        getattr(self, "run")()
    def order(self,fn):
        with open(fn, "r+") as f:
             lines = f.readlines()
             lines.sort()        
             f.seek(0)
             f.writelines(lines)
    def __init_(self):
        pass
    def run(self,strlinginput,stirlingoutput,temp_var=2):
        try:
          if(len(strlinginput) >=0 and len(stirlingoutput)>=0 and temp_var>=0):
              with open(strlinginput, 'r') as src:
                with open(stirlingoutput, 'w') as dest:
               
                   if((temp_var)%2==0):
                       pass
                   else:
                       temp_var+=1
                   i=0

                   for line in src:
                       if(i==temp_var-1):
                           dest.write('%s\t%s\n' % (str(i), line.rstrip('\n')))
                           i=0
                       else:
                           dest.write('%s\t%s\n' % (str(i), line.rstrip('\n')))
                           i+=1       
              self.order(stirlingoutput)
          else:
            pass
        except:
            print(usage)
if __name__ == "__main__":
	fp = add_prefix()
	fp.run(options.inp,options.out,options.nlb)
