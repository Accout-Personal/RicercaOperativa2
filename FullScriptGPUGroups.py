import json
import os
#import geometryUtil
import copy
#import networkx as nx
import csv
import numpy as np

import os
# needs to appear before `from numba import cuda`
#os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
## set to "1" for more debugging, but slower performance
#os.environ["NUMBA_CUDA_DEBUGINFO"] = "1"

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
def compare_groups_kernel(all_points_a, all_points_b, 
                         all_nfps, nfp_sizes, nfp_offsets,
                         batch_references,      # Add this - references for each group A in batch
                         pair_offsets_a, pair_offsets_b,
                         pair_sizes_a, pair_sizes_b,
                         num_pairs_in_batch,
                         results,max_results):
    """
    Process batch of pairs
    Memory optimized version using uint16 for counts and offsets
    NFPs handled with offsets and shifted by references
    """
    #One block handles 1 point to one group of points
    #Pair ID
    pair_idx = cuda.blockIdx.y
    if pair_idx >= num_pairs_in_batch:
        return
    
    #Block ID
    point_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if point_idx >= pair_sizes_a[pair_idx]:
        return
    
    # Get reference for this pair (based on group A)
    ref_x = batch_references[pair_idx, 0]
    ref_y = batch_references[pair_idx, 1]
    
    # Get our point from A
    point_a = all_points_a[pair_offsets_a[pair_idx] + point_idx]

    # Load NFP for this pair into shared memory using offsets and shift by reference
    shared_nfp = cuda.shared.array(shape=(MAX_NFP_VERTICES,2), dtype=np.float32)
    if cuda.threadIdx.x == 0:
        nfp_size = nfp_sizes[pair_idx]
        #print(nfp_size)
        nfp_start = nfp_offsets[pair_idx]
        #print("point a: ",point_a_x,point_a_y)
        for i in range(nfp_size):
            # Shift NFP by reference and point A
            shared_nfp[i,0] = all_nfps[nfp_start + i,0] - ref_x + point_a[0]
            shared_nfp[i,1] = all_nfps[nfp_start + i,1] - ref_y + point_a[1]
    
    cuda.syncthreads()
    
    # Process against all points in B group
    points_per_thread = (pair_sizes_b[pair_idx] + cuda.blockDim.x - 1) // cuda.blockDim.x
    start_b = cuda.threadIdx.x * points_per_thread
    end_b = min(start_b + points_per_thread, pair_sizes_b[pair_idx])
    
    b_offset = pair_offsets_b[pair_idx]
    result_offset = point_idx * pair_sizes_b[pair_idx]
    
    for j in range(pair_sizes_b[pair_idx]):
        point_b = all_points_b[b_offset + j]
        result_idx = point_idx * pair_sizes_b[pair_idx] + j
        
        if result_idx >= max_results:  #max_results to kernel
            return

        results[result_idx, 0] = point_a[2]
        results[result_idx, 1] = point_b[2]
        results[result_idx, 2] = point_in_polygon(point_b, shared_nfp, nfp_sizes[pair_idx])


    
def process_group_pairs(all_pairs,groups,group_poly_ids,nfps,nfp_sizes, references,batch_size=1):

    """
    Process multiple pairs in batches
    all_pairs: list of (group_a_idx, group_b_idx) to process
    groups: list of point arrays for each group
    nfps: dictionary or array of NFPs for each pair
    """

    # Prepare results dictionary
    results_list = []
    
    # Process in batches
    for batch_start in range(0, len(all_pairs), batch_size):
        #cuda.current_context().synchronize()
        batch_end = min(batch_start + batch_size, len(all_pairs))
        batch_pairs = all_pairs[batch_start:batch_end]
        num_pairs = len(batch_pairs)
        # Prepare concatenated arrays for this batch
        points_a = []
        points_b = []
        batch_nfps = []
        batch_nfp_sizes = []
        pair_offsets_a = np.zeros(num_pairs, dtype=np.uint32) # 0-65535 range
        pair_offsets_b = np.zeros(num_pairs, dtype=np.uint32)
        nfp_offsets = np.zeros(num_pairs, dtype=np.uint32)
        pair_sizes_a = np.zeros(num_pairs, dtype=np.uint32)
        pair_sizes_b = np.zeros(num_pairs, dtype=np.uint32)
        batch_references = np.zeros((num_pairs, 2), dtype=np.float32)  # 2D for x,y coordinates
        # Calculate offsets and concatenate data
        offset_a = 0
        offset_b = 0
        nfp_offset = 0
        print(f"Processing batch {batch_start//batch_size + 1}, pairs {batch_start} to {batch_end}")
        for i, (group_a_idx, group_b_idx) in enumerate(batch_pairs):
            #Get Referecens 
            batch_references[i] = references[group_poly_ids[group_a_idx]]
            # Get points for this pair
            group_a_points = groups[group_a_idx]['InnerFitPoints']
            group_b_points = groups[group_b_idx]['InnerFitPoints']
            
            # Store offsets and sizes
            pair_offsets_a[i] = offset_a
            pair_offsets_b[i] = offset_b
            pair_sizes_a[i] = len(group_a_points)
            pair_sizes_b[i] = len(group_b_points)
            
            # Concatenate points
            points_a.append(group_a_points)
            points_b.append(group_b_points)
            
            # Get NFP for this pair
            nfp = nfps[(group_poly_ids[group_a_idx], group_poly_ids[group_b_idx])]
            nfp_size = nfp_sizes[(group_poly_ids[group_a_idx], group_poly_ids[group_b_idx])]
            nfp_offsets[i] = nfp_offset
            nfp_offset += len(nfp)
            batch_nfps.append(nfp)
            batch_nfp_sizes.append(nfp_size)

            # Update offsets
            offset_a += len(group_a_points)
            offset_b += len(group_b_points)
            
        #print("nfp: ",nfp)
        #print("nfp sizes ",nfp_size)
        #print("references A",references)
        # Convert lists to arrays
        all_points_a = np.concatenate(points_a).astype(np.float32)
        all_points_b = np.concatenate(points_b).astype(np.float32)
        all_nfps = np.concatenate(batch_nfps).astype(np.float32)
        
        # Prepare results array - using uint32 for potentially millions of edges
        max_results = sum(int(pair_sizes_a[i]) * int(pair_sizes_b[i]) for i in range(num_pairs))

        batch_results = np.zeros((max_results, 3), dtype=np.uint32)
        
        # Transfer all data to GPU at once
        d_points_a = cuda.to_device(all_points_a)
        d_points_b = cuda.to_device(all_points_b)
        d_nfps = cuda.to_device(all_nfps)
        d_nfp_sizes = cuda.to_device(batch_nfp_sizes)
        d_nfp_offsets = cuda.to_device(nfp_offsets)
        d_references = cuda.to_device(batch_references)
        d_offsets_a = cuda.to_device(pair_offsets_a)
        d_offsets_b = cuda.to_device(pair_offsets_b)
        d_sizes_a = cuda.to_device(pair_sizes_a)
        d_sizes_b = cuda.to_device(pair_sizes_b)
        d_results = cuda.to_device(batch_results)
    
        # Configure kernel
        threadsperblock = 256
        max_points = int(max(pair_sizes_a))  # Convert uint16 to int for grid calculation
        blockspergrid = ((max_points + threadsperblock - 1) // threadsperblock, num_pairs)
        print("GPU starts")
        # Launch kernel for whole batch
        compare_groups_kernel[blockspergrid, threadsperblock](
            d_points_a, d_points_b,
            d_nfps, d_nfp_sizes,d_nfp_offsets,
            d_references,
            d_offsets_a, d_offsets_b,
            d_sizes_a, d_sizes_b,
            num_pairs,
            d_results,
            max_results
        )
        # Get results
        batch_results = d_results.copy_to_host()

        print("GPU ends")
        # Split results back into pairs
        result_offset = 0
        for i, (group_a_idx, group_b_idx) in enumerate(batch_pairs):
            size = int(pair_sizes_a[i]) * int(pair_sizes_b[i])
            #pair_results = batch_results[result_offset:result_offset + size]
            results_list.append(batch_results[result_offset:result_offset + size])
            result_offset += size
        #cuda.current_context().reset()
        print(f"Processed batch {batch_start//batch_size + 1}, pairs {batch_start} to {batch_end}")
    #print(results_list)
    return np.vstack(results_list)

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
    
    outputdir = dir_path+'/resultsGPU2/'+name
    
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

    num_polygon = len(polygons)


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
    layer = 0
    PolyIndex = 0
    PolyIndexDict = {}
    layer_poly_ids = {}
    maxnfp = 0              #Error check
    references = {}
    print("generating each layer of points..")
    
    AccIndex = 0
    for key, value in polygons.items():
        Nfpsdict = {}
        
        poly = key

        PolyIndexDict[PolyIndex] = poly
        
        for nfpoly in polygons[poly]['nfps']:
            Nfpsdict[nfpoly['POLYGON']] = nfpoly
            del Nfpsdict[nfpoly['POLYGON']]['POLYGON']
            if len(nfpoly['VERTICES']) > maxnfp:
                maxnfp = len(nfpoly['VERTICES'])
        polygons[poly]['nfpdict'] = Nfpsdict
        MainPiece = polygons[poly]
        innerfit = MainPiece["innerfit"]
        Nfps = MainPiece["nfps"]
        quantity = MainPiece.get("QUANTITY",1)
        print("polygon: ",poly, "Quantity:",quantity)
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
            for i in range(0,len(innerpoint)):
                innerpoint[i][2] = i+AccIndex
                
            AccIndex += len(innerpoint)
            #innerpoint = np.array(innerpoint,dtype=np.float32)
            #print("there is a",len(innerpoint)," of innerfit points! ")
            LayerOfpoint.append({"POLYGON":poly,"InnerFitPoints":innerpoint,"Layer":layer})
            LayerPoly.append((layer,poly))
            layer_poly_ids[layer] = PolyIndex
            layer += 1
        print(polygons[poly]['VERTICES'])
        references[PolyIndex] = (polygons[poly]['VERTICES'][0]['x'],polygons[poly]['VERTICES'][0]['y'])
        PolyIndex += 1

    #Check if max exceeds the the constant NFP
    if maxnfp > MAX_NFP_VERTICES:
        raise Exception(f"SIZE ERROR: nfp size ({maxnfp}) exceeds max size of the MAX_NFP_VERTICES({MAX_NFP_VERTICES})")

 
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

    num_groups = len(LayerOfpoint)
    # Create pairs to process (excluding self-pairs)
    all_pairs = [(i,j) for i in range(num_groups) 
                 for j in range(num_groups) if i != j]

    print(len(polygons))
    polygonPairs = [(i,j) for i in range(num_polygon)
                 for j in range(num_polygon)]
    
    print(polygonPairs)

    nfpPair = {}
    nfpPairSize = {}
    for i,j in polygonPairs:
        polyA = PolyIndexDict[i]
        polyB = PolyIndexDict[j]
        listpoint = []
        for v in polygons[polyA]['nfpdict'][polyB]['VERTICES']:
            listpoint.append([v['x'],v['y']])

        nfpPair[(i,j)] = np.array(listpoint,dtype=np.float32)
        nfpPairSize[(i,j)] = nfpPair[(i,j)].shape[0]

    print(layer_poly_ids)
    print("Generating NFP-graph..")
    results = process_group_pairs(all_pairs,LayerOfpoint,layer_poly_ids,nfpPair,nfpPairSize,references)
    print("NFP-graph generation complete!")
    print(results)
    print("result size pre-filter: ",results.shape[0])
    print("filtering the edges")
    results = results[results[:,2] != 0]
    print("filter complete!")
    print("after results: ",results.shape[0])
    #import code
    #code.interact(local=locals())

    ntXgraphAll.addEdges((np.array(results[:,0]),np.array(results[:,1])), addMissing=True,checkMultiEdge=True)
    ntXgraphInterLayer.addEdges((np.array(results[:,0]),np.array(results[:,1])), addMissing=True,checkMultiEdge=True)

    print("nXgraphAll node:",ntXgraphAll.numberOfNodes(),"edges: ",ntXgraphAll.numberOfEdges(),"cliques edges: ",ntXgraphComplete.numberOfEdges(),"NFP-edges: ",ntXgraphInterLayer.numberOfEdges(),"density:",nk.graphtools.density(ntXgraphAll))
    exit()
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

    print("nXgraphAll node:",ntXgraphAll.numberOfNodes(),"edges: ",ntXgraphAll.numberOfEdges(),"cliques edges: ",ntXgraphComplete.numberOfEdges(),"NFP-edges: ",ntXgraphInterLayer.numberOfEdges(),"density:",nk.graphtools.GraphTools.density(ntXgraphAll))

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
