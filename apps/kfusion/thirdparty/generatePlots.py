#!/usr/bin/python
# Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
# Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1
#
# This code is licensed under the MIT License.

import sys
import re
import math
import numpy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

kfusion_log_regex  =      "([0-9]+[\s]*)\\t" 
kfusion_log_regex += 8 *  "([0-9.]+)\\t" 
kfusion_log_regex += 3 *  "([-0-9.]+)\\t" 
kfusion_log_regex +=      "([01])\s+([01])" 

nuim_log_regex =      "([0-9]+)" 
nuim_log_regex += 7 * "\\s+([-0-9e.]+)\\s*" 


# open files

if len(sys.argv) != 3 :
    print "I need two parameters, the benchmark log file and the original scene camera position file."
    exit (1)

# open benchmark log file first
print "Get KFusion output data." 
framesDropped = 0
validFrames = 0
lastFrame = -1
untracked = -4;
kfusion_traj = []
fileref = open(sys.argv[1],'r')
data    = fileref.read()
fileref.close()
lines = data.split("\n") # remove head + first line
headers = lines[0].split("\t")
fulldata = {}
if len(headers) == 15 :
    if headers[14] == "" :
        del headers[14]
if len(headers) != 14 :
    print "Wrong KFusion log  file. Expected 14 columns but found " + str(len(headers))
    exit(1)
for variable in  headers[2:7] :
    fulldata[variable] = []

for line in lines[1:] :
    matching = re.match(kfusion_log_regex,line)
    if matching :
        dropped =  int( matching.group(1)) - lastFrame - 1
        if dropped>0:     		
    		framesDropped = framesDropped + dropped    		
    		for pad in range(0,dropped) :
    	         kfusion_traj.append( lastValid )     	            
                validFrames = validFrames +1
        for elem_idx in  range(len(headers)) :
            if headers[elem_idx] in fulldata:
                fulldata[headers[elem_idx]].append(float(matching.group(elem_idx+1)))
        
        lastFrame = int(matching.group(1))
    else :
        #print "Skip KFusion line : " + line
        break

       
#print "The following are designed to enable easy macchine readability of key data" 
#print "MRkey:,logfile,ATE,computaion,dropped,untracked"
#print ("MRdata:,%s,%6.6f,%6.6f,%d,%d") % ( sys.argv[1], numpy.mean(fulldata["ATE"]), numpy.mean(fulldata["computation"]), framesDropped, untracked)

print "\nGenerating plots.."

for variable in sorted(fulldata.keys()) :
    if "X" in variable or "Z" in variable or "Y" in variable or "frame" in variable  or "tracked" in variable      or "integrated" in variable  :  
        del fulldata[variable]
        continue

    if (framesDropped == 0)  and (str(variable) == "ATE_wrt_kfusion"):
        del fulldata[variable]
        continue
    if variable == "ATE":
        del fulldata[variable]
        continue
		
    print "%20.20s" % str(variable),
    print "\tMin : %6.6f"   % min(fulldata[variable]),
    print "\tMax : %0.6f"   % max(fulldata[variable]),
    print "\tMean : %0.6f"  % numpy.mean(fulldata[variable]),
    print "\tStdev : %0.6f" % numpy.std(fulldata[variable]),
    print "\tTotal : %0.8f" % sum(fulldata[variable])

# Configuring the plot

index = numpy.arange(len(fulldata.values()))
bar_width = 0.30
error_config = { 'ecolor': '0.3'} 
means = [numpy.mean(fulldata[variable]) for variable in fulldata.keys()]
stdv = [numpy.std(fulldata[variable]) for variable in fulldata.keys() ]
print "Dictionary length: ", index
plot = plt.bar(index + bar_width, means, bar_width, color='#43656a'  )

plt.xticks(index + (1.5 * bar_width), fulldata.keys(), rotation='vertical')
plt.grid(True)
plt.tight_layout()

pdf = PdfPages('plot.pdf')
plt.savefig(pdf, format='pdf')
pdf.close()

