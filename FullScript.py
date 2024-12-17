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

    blaz = "/datasetJson/json/blaz.json"
    shapes = "/datasetJson/json/shapes.json"
    shirts = "/datasetJson/json/shirts.json"
    dagli = "/datasetJson/json/dagli.json"
    fu = "/datasetJson/json/fu.json"
    fu5 = "/datasetJson/json/fu5.json"
    fu6 = "/datasetJson/json/fu6.json"
    fu7 = "/datasetJson/json/fu7.json"
    fu8 = "/datasetJson/json/fu8.json"
    fu9 = "/datasetJson/json/fu9.json"
    fu10 = "/datasetJson/json/fu10.json"
    jakobs1 = "/datasetJson/json/jakobs1.json"
    jakobs2 = "/datasetJson/json/jakobs2.json"
    poly1a = "/datasetJson/json/poly1a.json"
    poly1b = "/datasetJson/json/poly1b.json"
    poly1c = "/datasetJson/json/poly1c.json"
    poly1d = "/datasetJson/json/poly1d.json"
    poly1e = "/datasetJson/json/poly1e.json"
    three = "/datasetJson/json/three.json"
    rco = "/datasetJson/json/rco.json"
    #dataset struct:
    #{
    #   "name_polygon":{
    #       "VERTICES":[list_of_vertices],
    #       "QUANTITY":quantity,
    #       "NUMBER OF VERTICES": number
    #
    #}
    ##J


    #Dataset selected for graph generation
    selelected = three
    filepath = dir_path+selelected

    #Name of output

    name = Path(filepath).stem

    #Output directory
    
    outputdir = dir_path+'/results/'+name
    
    #Board Size
    width = 7 #y axis
    lenght = 7 #x axis
    PieceQuantity = 1 #None if use quantity from dataset
    with open(filepath) as file:
        inputdataset = json.load(file)
    subJackobs = {}
    #

    
    polygons = run_js_script_function('file://'+dir_path.replace('\\','/')+'/scriptpy.js', 'calcNFP_INFP', [inputdataset, lenght,width])
    
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

    print("nXgraphAll node:",ntXgraphAll.number_of_nodes(),"edges: ",ntXgraphAll.number_of_edges(),"cliques edges: ",ntXgraphComplete.number_of_edges(),"NFP-edges: ",ntXgraphInterLayer.number_of_edges(),"density:",nx.density(ntXgraphAll))

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

    
    #import code
    #code.interact(local=locals())
