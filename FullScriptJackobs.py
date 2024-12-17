import json
import os
import geometryUtil
import copy
import networkx as nx
import csv
from pathlib import Path
from NFPLib import *
if __name__ == "__main__":
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    jakobs1 = "/datasetJson/json/jakobs1.json"
    jakobs2 = "/datasetJson/json/jakobs2.json"
    #dataset struct:
    #{
    #   "name_polygon":{
    #       "VERTICES":[list_of_vertices],
    #       "QUANTITY":quantity,
    #       "NUMBER OF VERTICES": number
    #
    #}
    ##J
    Jackobs1 = {
        "J1_10_10_0":{"set":[1,5, 9, 12,15,16,17,20,21,23],"board":(10,20)},
        "J1_10_10_1":{"set":[1,3, 4, 5, 8, 9, 10,13,14,25],"board":(10,19)},
        "J1_10_10_2":{"set":[9,12,13,15,16,17,20,5, 2, 21],"board":(10,21)},
        "J1_10_10_3":{"set":[9,14,20,25,17,24,8, 22,6, 4],"board":(10,23)},
        "J1_10_10_4":{"set":[1,2, 3, 16,17,9, 22,10,8, 25],"board":(10,15)},

        "J1_12_20_0":{"set":[9,12,16,17,1,22,8,21,15,13,23,5],"board":(20,12)},
        "J1_12_20_1":{"set":[9,12,16,17,1,3,4,14,15,10,5,25],"board":(20,11)},
        "J1_12_20_2":{"set":[9,16,17,13,14,20,22,21,24,2,4,8],"board":(20,14)},
        "J1_12_20_3":{"set":[1,2,3,16,17,25,6,20,22,10,8,7],"board":(20,10)},
        "J1_12_20_4":{"set":[24,6,16,3,9,7,15,17,23,13,21,1],"board":(20,16)},

        "J1_14_20_0":{"set":[9,17,22,23,5,3,21,1,16,10,8,15,12,13],"board":(20,14)},
        "J1_14_20_1":{"set":[14,20,4,2,3,9,15,10,25,5,12,17,16,1],"board":(20,14)},
        "J1_14_20_2":{"set":[2,24,13,6,21,14,8,25,20,16,9,4,17,10],"board":(20,16)},
        "J1_14_20_3":{"set":[15,13,1,2,17,23,3,8,19,22,16,21,7,10],"board":(20,12)},
        "J1_14_20_4":{"set":[24,6,25,3,11,15,4,9,10,16,13,12,17,21],"board":(20,16)},

    }

    Jackobs2 = {
        "J2_10_35_0":{"set":[23,20,19,1,12,5,21,9,15,18],"board":(35,28)},
        "J2_10_35_1":{"set":[25,13,8,10,4,1,3,12,5,14],"board":(35,28)},
        "J2_10_35_2":{"set":[13,17,12,20,16,5,19,15,9,2],"board":(35,27)},
        "J2_10_35_3":{"set":[25,20,8,24,22,21,12,4,6,10],"board":(35,25)},
        "J2_10_35_4":{"set":[25,20,1,16,8,17,7,2,10,3],"board":(35,22)},

        "J2_12_35_0":{"set":[13,1,9,12,15,5,19,21,20,23,8,18],"board":(35,31)},
        "J2_12_35_1":{"set":[1,25,16,12,5,10,9,17,3,15,4,14],"board":(35,29)},
        "J2_12_35_2":{"set":[12,13,10,20,16,4,19,24,2,8,21,22],"board":(35,30)},
        "J2_12_35_3":{"set":[2,20,7,6,1,3,16,22,10,8,17,25],"board":(35,25)},
        "J2_12_35_4":{"set":[24,23,6,13,12,15,7,18,1,19,21,3],"board":(35,29)},

        "J2_14_35_0":{"set":[13,18,15,12,8,21,10,9,1,20,23,5,3,19],"board":(35,34)},
        "J2_14_35_1":{"set":[10,5,1,4,16,17,3,15,14,9,20,12,25,2],"board":(35,33)},
        "J2_14_35_2":{"set":[10,13,16,4,14,12,20,25,19,22,8,21,24,6],"board":(35,33)},
        "J2_14_35_3":{"set":[18,15,10,2,19,1,8,3,17,13,23,20,7,21],"board":(35,29)},
        "J2_14_35_4":{"set":[13,6,3,16,19,11,12,15,9,21,10,25,24,4],"board":(35,31)},
    }

    #Dataset selected for graph generation
    selelected = jakobs1
    filepath = dir_path+selelected
    with open(filepath) as file:
            inputdataset = json.load(file)

    metaResults = []
    for key,value in Jackobs2.items():
        
        #Name of output
        name = key

        #Output directory

        outputdir = dir_path+'/results/'+name

        #Board Size
        width = value['board'][0] #y axis
        lenght = value['board'][1] #x axis
        PieceQuantity = 1 #None if use quantity from dataset
        
        subJackobs = {}
        for poly in value['set']:
            subJackobs[str(poly)] = inputdataset[str(poly)]
        #


        polygons = run_js_script_function('file://'+dir_path.replace('\\','/')+'/scriptpy.js', 'calcNFP_INFP', [subJackobs, lenght,width])

        #nfp-infit output struct:
        #{
        #   "name_polygon":{
        #       "VERTICES":[list_of_vertices],
        #       "QUANTITY":quantity,
        #       "NUMBER OF VERTICES": number,
        #       "innerfit":[polygon_innerfit],
        #       "nfps":[{"POLYGON": "name_polygon","VERTICES":[list_of_vertices_nfp]}]}
        #
        #}
        #NOTE: There is also a 'rect' polygon added at the end of the file as the board

        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
        nfp_infp = json.dumps(polygons, indent=2)
        with open(outputdir+'/nfp-infp '+name+'.json', 'w') as file:
            file.write(nfp_infp)

        board = copy.deepcopy(polygons['rect'])
        del polygons['rect']

        print(len(polygons))
        total = 0
        if PieceQuantity is not None:
            for key, value in polygons.items():
                polygons[key]['QUANTITY'] = PieceQuantity
                total += polygons[key]['QUANTITY']
        else:
            for key, value in polygons.items():
                #polygons[key]['QUANTITY'] = PieceQuantity
                total += polygons[key]['QUANTITY']

        freq = 1 #gx=gy=1

        LayerPoly = []
        LayerOfpoint = []
        valIndex = 1
        layer = 1
        print("generating each layer of points..")
        for key, value in polygons.items():
            poly = key
            MainPiece = polygons[poly]
            innerfit = MainPiece["innerfit"]
            Nfps = MainPiece["nfps"]
            quantity = MainPiece.get("QUANTITY",1)
            print("polygon: ",key, "Quantity:",quantity)
            for i in range(quantity):
                boardPoints = generatePoints(board,freq,valIndex)
                valIndex += len(boardPoints)
                innerpoint = []
                dictBoardPolyinnerFit = dictpoly(innerfit)
                #Filter for innerfit for polygon
                for point in boardPoints:
                    res = geometryUtil.point_in_polygon(point,dictBoardPolyinnerFit)
                    if res != False:
                        innerpoint.append(point)
                #print("there is a",len(innerpoint)," of innerfit points! ")
                LayerOfpoint.append({"POLYGON":key,"InnerFitPoints":innerpoint,"Layer":layer})
                LayerPoly.append((layer,key))
                layer += 1



        print("generating the graph..")
        ntXgraphAll = nx.Graph()
        ntXgraphInterLayer = nx.Graph()
        ntXgraphComplete = nx.Graph()
        for mainLayer in LayerOfpoint:
            LayerGraph = {}
            #make full graph
            print("generating graph of Layer: ",mainLayer["Layer"])
            completeGraph = makeFullGraph(mainLayer["InnerFitPoints"])
            for g in completeGraph:
                ntXgraphAll.add_edge(g[0],g[1])
                ntXgraphComplete.add_edge(g[0],g[1])

            #FinalGraph.extend(completeGraph)
            LayerGraph['completeGraph'] = completeGraph
            LayerGraph["Layer"] = mainLayer["Layer"]

            MainPoly = mainLayer['POLYGON']
            print("iterating point on Layer: ",mainLayer["Layer"])
            resList = MakePointGraph_process_pool(mainLayer["InnerFitPoints"],LayerOfpoint,mainLayer,polygons,dictBoardPolyinnerFit)
            for res in resList[0]:
                for p in res:
                    ntXgraphAll.add_edge(p['idA'],p['idB'])
                    ntXgraphInterLayer.add_edge(p['idA'],p['idB'])

        print("write into file ...")


        with open(outputdir+'/pointCoordinate '+name+'.txt', 'w') as file:
            file.write('##format: (Layer,x,y,id)' + '\n')
            for layer in LayerOfpoint:
                for points in layer['InnerFitPoints']:
                    file.write(str(layer['Layer'])+' '+str(points['x'])+' '+str(points['y'])+' '+str(points['id']) + '\n')

        with open(outputdir+'/LayerPoly'+name+'.txt', 'w') as file:
            file.write('##format:  Layer polygon ' + '\n')
            for layer in LayerPoly:
                file.write(str(layer[0])+'\t'+str(layer[1])+'\n')


        with open(outputdir+'/graph '+name+'.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter='\t',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for edge in list(ntXgraphAll.edges()):
                spamwriter.writerow([edge[0],edge[1]])

        print("node:",ntXgraphAll.number_of_nodes(),"edges: ",ntXgraphAll.number_of_edges(),"cliques edges: ",ntXgraphComplete.number_of_edges(),"NFP-edges: ",ntXgraphInterLayer.number_of_edges(),"density:",nx.density(ntXgraphAll))
        metaResults.append({ 
                "Name :":str(name),
                "Total Pieces :":total,
                "Type of Pieces :":len(polygons),
                "lenght x lenght:": str(width)+' '+ str(lenght),
                "Nodes:":ntXgraphAll.number_of_nodes(),
                "Edges:":ntXgraphAll.number_of_edges(),
                "Clique Edges:":ntXgraphComplete.number_of_edges(),
                "NFP-Edges:":ntXgraphInterLayer.number_of_edges()
            })
        

        with open(outputdir+'/metadata'+name+'.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter='\t',
                                    quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(["Name :",str(name)])
            spamwriter.writerow(["Total Pieces :",total])
            spamwriter.writerow(["Type of Pieces :",len(polygons)])
            spamwriter.writerow(["Board lenght x lenght:", str(width)+' '+ str(lenght)])
            spamwriter.writerow(["Number of Nodes:",ntXgraphAll.number_of_nodes()])
            spamwriter.writerow(["Number of Edges:",ntXgraphAll.number_of_edges()])
            spamwriter.writerow(["Number of Clique Edges:",ntXgraphComplete.number_of_edges()])
            spamwriter.writerow(["Intra Layer Edges:",ntXgraphInterLayer.number_of_edges()])

    for r in metaResults:
        strs = ''
        for k,v in r.items():
           strs = strs+str(k)+' '+str(v)+' '
        print(strs)
    
    import code
    code.interact(local=locals())
