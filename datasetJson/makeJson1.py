import os
import json
dir_path = os.path.dirname(os.path.realpath(__file__))

blaz = "/blaz_2007-04-23/blaz.txt"
shapes = "/shapes_2007-04-23/shapes.txt"
shirts = "/shirts_2007-05-15/shirts.txt"

filepath = dir_path+shirts
datasetDict = {}
datasetDict
with open(filepath,'r') as lines:
    prev=""
    currentpoly=""

    for line in lines:
        #print(line)
        line = line.strip()
        if line == 'QUANTITY':
           
            currentpoly=prev
            print("new polygon:",currentpoly)
            #new polygon
            datasetDict[currentpoly] = {}
            datasetDict[currentpoly]['VERTICES'] = []
        elif line.isnumeric() and prev == 'QUANTITY':
            print("quantity for polygon",currentpoly)
            datasetDict[currentpoly]['QUANTITY'] = int(line)
        elif line.isnumeric() and prev == 'NUMBER OF VERTICES':
            print("quantity of vertex")
            datasetDict[currentpoly]['NUMBER OF VERTICES'] = int(line)
        else:
            splitted = line.split(" ")
            if len(splitted) == 1 and splitted[0] == '': #empty row -->end polygon
                print("empty row")
                prev=""
                currentpoly=""
                continue
            elif len(splitted) >= 3: #coordinate x y
                numbers = list(filter(lambda x : x.lstrip("-").isnumeric(),splitted)) # filter out spaces and words
                print(numbers)
                if len(numbers) == 2:
                    try:
                        x = int(numbers[0])
                    except ValueError: #in case of float
                        x = float(numbers[0])

                    try:
                        y = int(numbers[1])
                    except ValueError: #in case of float
                        y = float(numbers[1])
                    datasetDict[currentpoly]["VERTICES"].append({"x":x,"y":y})
        prev = line

print(datasetDict)

# Serializing json
json_object = json.dumps(datasetDict, indent=4)
 
# Writing to sample.json
with open((filepath[:-3]+'json'), "w") as outfile:
    outfile.write(json_object)