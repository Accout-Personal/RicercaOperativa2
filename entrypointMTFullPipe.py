import json
import os
import math
import geometryUtil
import copy
import concurrent.futures
import multiprocessing
from tqdm import tqdm
import logging
from datetime import datetime
import networkx as nx
import csv
import subprocess
from pathlib import Path

###MULTITHREADING###
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f'processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
)

#def process_item(item):
#    """Example processing function - replace with your actual processing logic"""
#    try:
#        time.sleep(0.1)  # Simulate some work
#        return f"Processed {item}"
#    except Exception as e:
#        logging.error(f"Error processing item {item}: {str(e)}")
#        return None

def MakePointGraph_process_pool(listOfPoints,LayerOfpoint,MainLayer,polygons,dictBoardPoly, max_workers=None, chunk_size=1000):
    """
    Process items using ProcessPoolExecutor
    Good for CPU bound tasks (e.g., heavy computations, data processing)
    """
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    results = []
    errors = []
    
    logging.info(f"Starting process pool processing with {max_workers} workers")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and process them with progress bar
        futures = []
        for point in listOfPoints:
            future = executor.submit(MakePointGraph, point,MainLayer,LayerOfpoint,polygons,dictBoardPoly)
            futures.append(future)
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                errors.append(str(e))
                logging.error(f"Task failed: {str(e)}")
    
    logging.info(f"Processing completed. Successfully processed: {len(results)}, Errors: {len(errors)}")
    return results, errors

###END MULTITHREADING###

def MakePointGraph(point,mainLayer,LayerOfpoint,polygons,dictBoardPoly):
    #LayerGraph = {}
    pointGraph = []
    Allnfps = polygons[mainLayer["POLYGON"]]['nfps']
    for otherLayer in LayerOfpoint:
        #only performs connection to other layer
        
        if otherLayer["Layer"] != mainLayer['Layer']:
            #print("iterating layers, comparing layer from ",mainLayer['Layer'], "to Layer: ",otherLayer["Layer"],"point id: ",point["id"])
            #find the nfp main-other
            
            nftpVerices = []
            for nfp in Allnfps:
                if nfp["POLYGON"] == otherLayer['POLYGON']:
                    nftpVerices = copy.deepcopy(nfp["VERTICES"]) #copy, otherwise assignment is by reference.
            
            #calculate the offset operation to the nfp polygon relative to the reference point (first point of polygon) and board point
            offsetx = point['x']-polygons[mainLayer["POLYGON"]]['VERTICES'][0]['x']
            offsety = point['y']-polygons[mainLayer["POLYGON"]]['VERTICES'][0]['y']
            #make offset the nfp (operation by reference)
            for v in nftpVerices:
                v['x'] = v['x'] + offsetx
                v['y'] = v['y'] + offsety
            nftpVerices = dictpoly(nftpVerices)
            ##get point of no-fit polygon of second layer and make edge
            #LayerGraph[otherLayer["Layer"]] = []
            
            for secondaryPoint in otherLayer["InnerFitPoints"]:
                res = geometryUtil.point_in_polygon(secondaryPoint,nftpVerices)
                if res != False and res is not None: #exclude the border case
                    edge = {
                        'layerA':mainLayer['Layer'],
                        'xA':point['x'],
                        'yA':point['y'],
                        'idA':point['id'],

                        'layerB':otherLayer["Layer"],
                        'xB':secondaryPoint['x'],
                        'yB':secondaryPoint['y'],
                        'idB':secondaryPoint['id']}
                    #LayerGraph[otherLayer["Layer"]].append(edge)
                    pointGraph.append(edge)
    #return (pointGraph,LayerGraph)
    return pointGraph

def makeFullGraph(PointsList):
    edgeList = []
    for i in range(len(PointsList)):
        for j in range(i+1,len(PointsList)):
            edgeList.append((PointsList[i]['id'],PointsList[j]['id']))
    return edgeList

def dictpoly(poly):
    dictPoly = {}
    for i in range(len(poly)):
        dictPoly[i] = poly[i]
    dictPoly['len'] = len(poly)
    return dictPoly 

def generatePoints(board,freq,startIndex=1):
    minx = miny = math.inf
    maxx = maxy = -math.inf
    for point in board:
        
        if point['x'] < minx:
            minx = point['x']
        elif point['x'] > maxx:
            maxx = point['x']
        
        if point['y'] < miny:
            miny = point['y']
        elif point['y'] > maxy:
            maxy = point['y']
    if minx == math.inf:
        raise TypeError("error: no min x found!")    
    if miny == math.inf:
        raise TypeError("error: no min y found!")

    if maxx == -math.inf:
        raise TypeError("error: no max x found!")

    if maxy == -math.inf:
        raise TypeError("error: no max y found!")
      
    stepsx = int((maxx-minx)/freq)
    stepsy = int((maxy-miny)/freq)
    points = []
    acc = startIndex
    for i in range(0,stepsx):
        for j in range(0,stepsy):
            points.append({"x":minx+(i*freq),"y":miny+(j*freq),"id":acc})
            acc+=1
    
    return points

def run_js_script_function(script_path, function_name, args=None):
    # First create a setup file that defines all globals
    setup_code = """
    globalThis.self = globalThis;
    globalThis.window = globalThis;
    global.navigator = {
        userAgent: 'node',
        platform: 'node',
        language: 'en',
        languages: ['en'],
        onLine: true,
        // Add more navigator properties as needed
    };
    """
    
    with open('setup.mjs', 'w') as f:
        f.write(setup_code)
    
    # Then create the main wrapper that imports setup first
    js_wrapper = f"""
    import './setup.mjs';
    import * as module from '{script_path}';
    
    const result = module.{function_name}({', '.join(str(arg) for arg in args) if args else ''});
    // Add a marker for our actual result
    console.log('RESULT_START');
    console.log(JSON.stringify(result));
    console.log('RESULT_END');
    """
    
    with open('temp_wrapper.mjs', 'w') as f:
        f.write(js_wrapper)
    
    try:
        process = subprocess.Popen(
            ['node', '--experimental-specifier-resolution=node', 'temp_wrapper.mjs'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        output, errors = process.communicate()
        
        # Cleanup
        os.remove('setup.mjs')
        os.remove('temp_wrapper.mjs')
        
        if process.returncode == 0:
            # Find the result between our markers
            lines = output.strip().split('\n')
            start_idx = -1
            end_idx = -1
            for i, line in enumerate(lines):
                if line == 'RESULT_START':
                    start_idx = i
                elif line == 'RESULT_END':
                    end_idx = i
            
            if start_idx != -1 and end_idx != -1:
                result_line = lines[start_idx + 1]
                return json.loads(result_line)
            else:
                raise Exception("Couldn't find marked result in output")
        else:
            raise Exception(f"JavaScript Error: {errors.strip()}")
            
    except FileNotFoundError:
        # Cleanup in case of error
        if os.path.exists('setup.mjs'): os.remove('setup.mjs')
        if os.path.exists('temp_wrapper.mjs'): os.remove('temp_wrapper.mjs')
        raise Exception("Node.js is not installed or not in PATH")


if __name__ == "__main__":
    


    dir_path = os.path.dirname(os.path.realpath(__file__))

    blaz = "/datasetJson/blaz.json"
    shapes = "/datasetJson/shapes.json"
    shirts = "/datasetJson/shirts.json"
    dagli = "/datasetJson/dagli.json"
    fu = "/datasetJson/fu.json"
    jakobs1 = "/datasetJson/jakobs1.json"
    jakobs2 = "/datasetJson/jakobs2.json"
    poly1a = "/datasetJson/poly1a.json"
    poly2b = "/datasetJson/poly2b.json"
    poly3b = "/datasetJson/poly3b.json"
    poly4b = "/datasetJson/poly4b.json"
    poly5b = "/datasetJson/poly5b.json"
    three = "/datasetJson/three.json"
    rco = "/datasetJson/rco.json"
    #dataset struct:
    #{
    #   "name_polygon":{
    #       "VERTICES":[list_of_vertices],
    #       "QUANTITY":quantity,
    #       "NUMBER OF VERTICES": number
    #
    #}

    #Dataset selected for graph generation
    selelected = three
    filepath = dir_path+selelected

    #Name of output

    name = Path(filepath).stem

    #Output directory
    
    outputdir = dir_path+'/results/'+name
    
    #Board Size
    width = 60 #y axis
    lenght = 25 #x axis
    PieceQuantity = 1
    with open(filepath) as file:
        inputdataset = json.load(file)
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
    for key, value in polygons.items():
        polygons[key]['QUANTITY'] = PieceQuantity
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

    print("nXgraphAll node:",ntXgraphAll.number_of_nodes(),"edges: ",ntXgraphAll.number_of_edges(),"density:",nx.density(ntXgraphAll))

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
