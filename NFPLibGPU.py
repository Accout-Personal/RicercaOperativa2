import math
import concurrent.futures
import multiprocessing
from tqdm import tqdm
import logging
import subprocess
from datetime import datetime
import copy
import geometryUtil, os ,json
import numpy as np

def MakePointGraph(point,mainLayer,LayerOfpoint,polygons):
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
    # Assuming PointsList is your original list of dictionaries
    # First extract the values we need into a numpy array
    points_values = np.array([point[2] for point in PointsList], dtype=int)

    # Generate all pairs of indices
    idx_i, idx_j = np.triu_indices(len(points_values), k=1)

    # Create the return array directly
    returnArr = np.column_stack((points_values[idx_i], points_values[idx_j]))
    return returnArr


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

