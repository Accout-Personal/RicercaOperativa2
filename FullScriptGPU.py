import json
import os
import copy
import networkx as nx
import csv
import numpy as np
from numba import cuda
from pathlib import Path
from NFPLibGPU import *
from typing import Tuple
import gc
import networkit as nk


def filteredges(point):
    return point[2] != 0
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
    selelected = rco
    filepath = dir_path+selelected

    #Name of output

    name = Path(filepath).stem+'1'

    #Output directory
    
    outputdir = dir_path+'/resultsGPU/'+name
    
    #Board Size
    width = 15 #y axis
    lenght = 8 #x axis
    PieceQuantity = 1 #None if use quantity from dataset
    freq = 1 #gx=gy=1
    with open(filepath) as file:
        inputdataset = json.load(file)
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



    LayerPoly = []
    LayerOfpoint = []
    valIndex = 1
    layer = 1
    pointIndex = 0
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
                    innerpoint.append([point['x'],point['y'],point['id']])
                for i in range(len(innerpoint)):
                    innerpoint[i][2] = i+pointIndex
            pointIndex += len(innerpoint)
            #innerpoint = np.array(innerpoint,dtype='float64')
            #print("there is a",len(innerpoint)," of innerfit points! ")
            LayerOfpoint.append({"POLYGON":key,"InnerFitPoints":innerpoint,"Layer":layer})
            LayerPoly.append((layer,key))
            layer += 1

    #print(LayerOfpoint[0]['InnerFitPoints'].shape)
    #print(LayerOfpoint[0]['InnerFitPoints'].dtype)
    #print(type(LayerOfpoint[0]['InnerFitPoints']))
    #print(len(LayerOfpoint[0]['InnerFitPoints']))
 
    print("generating the graph..\n")
    ntXgraphAll = nk.Graph()
    ntXgraphInterLayer = nk.Graph()
    ntXgraphComplete = nk.Graph()
    EdgeArray = []

    print("Generating complete graph..")
    for mainLayer in LayerOfpoint:
        #LayerGraph = {}
        #make full graph
        print("generating graph of Layer: ",mainLayer["Layer"])
        newArr = makeFullGraph(mainLayer["InnerFitPoints"])
        EdgeArray.append(newArr)

    EdgeArray = np.vstack(EdgeArray)
    print(EdgeArray[0:10])
    print(EdgeArray[:,0][0:10])
    print("adding results to the graph..")
    ntXgraphAll.addEdges((np.array(EdgeArray[:,0]),np.array(EdgeArray[:,1])), addMissing=True)
    ntXgraphComplete.addEdges((np.array(EdgeArray[:,0]),np.array(EdgeArray[:,1])), addMissing=True)
    print("adding complete!")

    print("Generating NFP-graph..")
    print(len(LayerOfpoint))
    totalNPFs = []
    for i in range(len(LayerOfpoint)-1):

        for j in range(i,len(LayerOfpoint)):
            
            if LayerOfpoint[i]['Layer'] !=  LayerOfpoint[j]['Layer']:
                print("comparing between: ",LayerOfpoint[i]['Layer']," ",LayerOfpoint[j]['Layer'])
                polyA = LayerOfpoint[i]['POLYGON']
                polyB = LayerOfpoint[j]['POLYGON']

                nfp_ab = None
                for nfp in polygons[polyA]['nfps']:
                    if nfp['POLYGON'] == polyB:
                        nfp_ab=[]
                        for v in nfp['VERTICES']:
                            nfp_ab.append([v['x'],v['y']])
                        break

                nfp_ba = None
                for nfp in polygons[polyB]['nfps']:
                    nfp_ba=[]
                    if nfp['POLYGON'] == polyA:
                        for v in nfp['VERTICES']:
                            nfp_ba.append([v['x'],v['y']])
                        break
                
                if nfp_ab is None or len(nfp_ab)==0 or nfp_ba is None  or len(nfp_ba)==0:
                    raise("Error nfp not found!")

                nfp_ab = np.array(nfp_ab,dtype='float32')
                nfp_ba = np.array(nfp_ba,dtype='float32')
                refa = np.array([polygons[polyA]['VERTICES'][0]['x'],polygons[polyA]['VERTICES'][0]['y']],dtype='float32')
                refb = np.array([polygons[polyB]['VERTICES'][0]['x'],polygons[polyB]['VERTICES'][0]['y']],dtype='float32')
                res_ab,res_ba = process_group_pair(LayerOfpoint[i]["InnerFitPoints"],LayerOfpoint[j]["InnerFitPoints"],nfp_ab,nfp_ba,refa,refb)

                res_ab = res_ab.reshape(-1, *res_ab.shape[2:])
                res_ab = res_ab[res_ab[:,2] ==1 ]
                
                #ntXgraphAll.addEdges((np.array(res_ab[:,0]),np.array(res_ab[:,1])), addMissing=True)
                #ntXgraphInterLayer.addEdges((np.array(res_ab[:,0]),np.array(res_ab[:,1])), addMissing=True)

                res_ba = res_ba.reshape(-1, *res_ba.shape[2:])
                res_ba = res_ba[res_ba[:,2] ==1 ]
                totalNPFs.append(res_ab)
                totalNPFs.append(res_ba)
                #ntXgraphAll.addEdges((np.array(res_ba[:,0]),np.array(res_ba[:,1])), addMissing=True)
                #ntXgraphInterLayer.addEdges((np.array(res_ba[:,0]),np.array(res_ba[:,1])), addMissing=True)

                gc.collect()
    totalNPFs = np.vstack(totalNPFs)
    print("graph generation complete!")
    print(totalNPFs)
    ntXgraphAll.addEdges((np.array(totalNPFs[:,0]),np.array(totalNPFs[:,1])), addMissing=True,checkMultiEdge=True)
    ntXgraphInterLayer.addEdges((np.array(totalNPFs[:,0]),np.array(totalNPFs[:,1])), addMissing=True,checkMultiEdge=True)
    #exit()
    
    print("writting into file ...")

    
    with open(outputdir+'/pointCoordinate '+name+'.txt', 'w') as file:
        file.write('##format: (Layer,x,y,id)' + '\n')
        for layer in LayerOfpoint:
            for points in layer['InnerFitPoints']:
                file.write(str(layer['Layer'])+' '+str(points[0])+' '+str(points[1])+' '+str(points[2]) + '\n')

    with open(outputdir+'/LayerPoly'+name+'.txt', 'w') as file:
        file.write('##format:  Layer polygon ' + '\n')
        for layer in LayerPoly:
            file.write(str(layer[0])+'\t'+str(layer[1])+'\n')
    

    with open(outputdir+'/graph '+name+'.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for edge in list(ntXgraphAll.iterEdges()):
            spamwriter.writerow([edge[0],edge[1]])

    print("nXgraphAll node:",ntXgraphAll.numberOfNodes(),"edges: ",ntXgraphAll.numberOfEdges(),"cliques edges: ",ntXgraphComplete.numberOfEdges(),"NFP-edges: ",ntXgraphInterLayer.numberOfEdges(),"density:",nk.graphtools.density(ntXgraphAll))

    with open(outputdir+'/metadata'+name+'.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t',
                                quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(["Name :",str(name)])
        spamwriter.writerow(["Total Pieces :",total])
        spamwriter.writerow(["Type of Pieces :",len(polygons)])
        spamwriter.writerow(["Board lenght x lenght:", str(width)+' '+ str(lenght)])
        spamwriter.writerow(["Number of Nodes:",ntXgraphAll.numberOfNodes()])
        spamwriter.writerow(["Number of Edges:",ntXgraphAll.numberOfEdges()])
        spamwriter.writerow(["Number of Clique Edges:",ntXgraphComplete.numberOfEdges()])
        spamwriter.writerow(["Intra Layer Edges:",ntXgraphInterLayer.numberOfEdges()])

    
    #import code
    #code.interact(local=locals())
