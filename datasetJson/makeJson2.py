import os
import json
dir_path = os.path.dirname(os.path.realpath(__file__))

dagli = "/dagli_2007-05-15/dagli.txt"
fu = "/fu_2007-05-15/fu.txt"
jakobs1 = "/jakobs_2007-04-23/jakobs1.txt"
jakobs2 = "/jakobs_2007-04-23/jakobs2.txt"
poly1a = "/poly/poly1a.txt"
poly2b = "/poly/poly2b.txt"
poly3b = "/poly/poly3b.txt"
poly4b = "/poly/poly4b.txt"
poly5b = "/poly/poly5b.txt"
all = [dagli,fu,jakobs1,jakobs2,poly1a,poly2b,poly3b,poly4b,poly5b]
for dir in all:
    filepath = dir_path+dir
    datasetDict = {}
    fileList = []
    with open(filepath,'r') as lines:
        for line in lines:
            fileList.append(line.strip())
    for i in range(0,len(fileList),2):
        tokens1 = fileList[i].split("\t")
        tokens2 = fileList[i+1].split("\t")[1:]
        polygon = tokens1[0]
        tokens1 = tokens1[2:]
        datasetDict[polygon] = {}
        datasetDict[polygon]["VERTICES"] = []
        datasetDict[polygon]["NUMBER OF VERTICES"] = len(tokens2)
        datasetDict[polygon]["QUANTITY"] = 1
        for j in range(len(tokens1)):
            try:
                x = int(tokens1[j])
            except ValueError:
                x = float(tokens1[j])
            
            try:
                y = int(tokens2[j])
            except ValueError:
                y = float(tokens2[j])
                
            datasetDict[polygon]["VERTICES"].append({"x":x,"y":y})

    # Serializing json
    json_object = json.dumps(datasetDict, indent=4)
    
    # Writing to sample.json
    with open((filepath[:-3]+'json'), "w") as outfile:
        outfile.write(json_object)