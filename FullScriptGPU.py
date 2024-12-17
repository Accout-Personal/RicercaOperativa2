import json
import os
#import geometryUtil
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
MAX_NFP_VERTICES = 30

@cuda.jit(device=True)
def on_segment(Ax,Ay, Bx,By, p):
    #p[0] => p['x'], p[1]=>p['y']
    # vertical line
    if almost_equal(Ax, Bx) and almost_equal(p[0], Ax):
        if not almost_equal(p[1], By) and not almost_equal(p[1], Ay) and p[1] < max(By, Ay) and p[1] > min(By, Ay):
            return True
        else:
            return False

    # horizontal line
    if almost_equal(Ay, By) and almost_equal(p[1], Ay):
        if not almost_equal(p[0], Bx) and not almost_equal(p[0], Ax) and p[0] < max(Bx, Ax) and p[0] > min(Bx, Ax):
            return True
        else:
            return False

    # range check
    if (p[0] < Ax and p[0] < Bx) or (p[0] > Ax and p[0] > Bx) or (p[1] < Ay and p[1] < By) or (p[1] > Ay and p[1] > By):
        return False

    # exclude end points
    if (almost_equal(p[0], Ax) and almost_equal(p[1], Ay)) or (almost_equal(p[0], Bx) and almost_equal(p[1], By)):
        return False

    cross = (p[1] - Ay) * (Bx - Ax) - (p[0] - Ax) * (By - Ay)

    if abs(cross) > 1e-9:
        return False

    dot = (p[0] - Ax) * (Bx - Ax) + (p[1] - Ay) * (By - Ay)

    if dot < 0 or almost_equal(dot, 0):
        return False

    len2 = (Bx - Ax)**2 + (By - Ay)**2

    if dot > len2 or almost_equal(dot, len2):
        return False

    return True

@cuda.jit(device=True)
def almost_equal(a, b):
    return abs(a - b) < 1e-9


@cuda.jit(device=True)
def point_in_polygon(point, polygon,number_vertices):
    
    # return True if point is in the polygon, False if outside, and None if exactly on a point or edge
    #False =>0, True =>1, None=>2
    inside = 0

    offsetx = 0
    offsety = 0
    #0 => x 1=>y
    j = number_vertices - 1
    for i in range(number_vertices):
        xi = polygon[i][0] + offsetx
        yi = polygon[i][1] + offsety
        xj = polygon[j][0] + offsetx
        yj = polygon[j][1] + offsety

        if almost_equal(xi, point[0]) and almost_equal(yi, point[1]):
            return 2  # no result
        
        if on_segment(xi, yi, xj, yj, point):
            return 2  # exactly on the segment
        
        if almost_equal(xi, xj) and almost_equal(yi, yj):  # ignore very small lines
            continue
        
        intersect = ((yi > point[1]) != (yj > point[1])) and (point[0] < (xj - xi) * (point[1] - yi) / (yj - yi) + xi)
        if intersect:
            #inside = not inside
            if inside==0:
                inside=1
            else:
                inside=0
        
        j = i
    #print("point in polygon check")
    #print(polygon)
    #print(point)
    #print(inside)
    return inside

@cuda.jit
def compare_groups_kernel(points_a, points_b, 
                         ab_nfp, ab_nfp_num_vertices,
                         results_ab,reference_a):
    """
    Kernel to compare points between two groups using polygon interactions
    Each thread handles one point from A against all points from B
    """
    # Block handles one point from A
    point_a_idx = cuda.blockIdx.x
    if point_a_idx >= points_a.shape[0]:
        return
    
    
    # Thread handles subset of points from B
    thread_idx = cuda.threadIdx.x
    # Shared memory for interaction polygons
    shared_ab_nfp = cuda.shared.array(shape=(MAX_NFP_VERTICES,2), dtype=np.float32)
    # Get our point from group A
    point_a = points_a[point_a_idx]

    #load polygon vertices and shifts with offset
    if cuda.threadIdx.x == 0:  #thread 0 load shared memory        
        for i in range(ab_nfp_num_vertices):
            shared_ab_nfp[i,0] = ab_nfp[i,0] - reference_a[0] + point_a[0] #xoffset
            shared_ab_nfp[i,1] = ab_nfp[i,1] - reference_a[1] + point_a[1] #yoffset
    
    cuda.syncthreads()
    
    # Each thread handles a portion of B points
    points_per_thread = (points_b.shape[0] + cuda.blockDim.x - 1) // cuda.blockDim.x
    start_idx = thread_idx * points_per_thread
    end_idx = min(start_idx + points_per_thread, points_b.shape[0])

    # Process our portion of B points and write results directly
    for j in range(start_idx, end_idx):
        point_b = points_b[j]
        results_ab[point_a_idx, j,0] = point_a[2]
        results_ab[point_a_idx, j,1] = point_b[2]
        results_ab[point_a_idx, j,2] = point_in_polygon(point_b, shared_ab_nfp, ab_nfp_num_vertices)

    
def process_group_pair(group_a, group_b, ab_nfp, ba_nfp,ref_a,ref_b):

    group_a = np.array(group_a, dtype=np.float32)
    group_b = np.array(group_b, dtype=np.float32)
    ab_nfp = np.array(ab_nfp, dtype=np.float32)
    ba_nfp = np.array(ba_nfp, dtype=np.float32)
    ref_a = np.array(ref_a, dtype=np.float32)
    ref_b = np.array(ref_b, dtype=np.float32)

    """Process a pair of groups in both directions"""
    # Unpack interaction data
    ab_nfp, ab_num_vertices = ab_nfp,ab_nfp.shape[0]
    ba_nfp, ba_num_vertices = ba_nfp,ba_nfp.shape[0]
    
    # Prepare results arrays
    results_ab = np.zeros((len(group_a), len(group_b), 3), dtype=np.int32)
    results_ba = np.zeros((len(group_b), len(group_a), 3), dtype=np.int32)
    
    # Configure kernel
    threadsperblock = 256
    blockspergrid_ab = len(group_a)  # One block per point in A

    d_points_a = cuda.to_device(group_a)
    d_points_b = cuda.to_device(group_b)
    d_ab_vertices = cuda.to_device(ab_nfp)
    d_results_ab = cuda.to_device(results_ab)
    #test_kernel[blockspergrid_ab, threadsperblock](d_points_a,d_points_b,d_ab_vertices,ab_num_vertices,d_results_ab,ref_a)

    compare_groups_kernel[blockspergrid_ab, threadsperblock](
        d_points_a, d_points_b,
        d_ab_vertices, ab_num_vertices,
        d_results_ab,ref_a
    )
    # Process B->A
    blockspergrid_ba = len(group_b)  # One block per point in A
    
    d_ba_vertices = cuda.to_device(ba_nfp)
    d_results_ba = cuda.to_device(results_ba)

    compare_groups_kernel[blockspergrid_ba, threadsperblock](
        d_points_b, d_points_a,  # Note: swapped order
        d_ba_vertices, ba_num_vertices,
        d_results_ba,ref_b
    )
    
    # Get results
    results_ab = d_results_ab.copy_to_host()
    results_ba = d_results_ba.copy_to_host()
    
    return results_ab, results_ba

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
    selelected = three
    filepath = dir_path+selelected

    #Name of output

    name = Path(filepath).stem

    #Output directory
    
    outputdir = dir_path+'/resultsGPU/'+name
    
    #Board Size
    width = 7 #y axis
    lenght = 7 #x axis
    PieceQuantity = 1 #None if use quantity from dataset
    freq = 1 #gx=gy=1
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
                    innerpoint.append([point['x'],point['y'],point['id']])
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
