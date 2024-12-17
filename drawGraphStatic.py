import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os ,copy ,json
def process_line(shared_dict,line):
    if line[0] =='#':
         return
    else:
        line = line.strip()
        s = line.split(' ')
        shared_dict['G'].add_edge(int(s[0]),int(s[1]))

def drawGraph(G,pos,Selnode,neighbors=False):
    
    # Get neighbors of node Selnode
    if neighbors:
        neighbors = list(G.neighbors(Selnode))
        node_colors = ['red' if node in neighbors else 'blue' if node == Selnode else 'lightblue' for node in G.nodes()]
    else:
        node_colors = ['blue' if node == Selnode else 'lightblue' for node in G.nodes()]
    plt.figure(figsize=(16, 12))
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    

    nx.draw_networkx_nodes(G, pos,
                          node_color=node_colors,
                          node_size=20)

    plt.title('Nodes with Coordinate Axes')

def drawPolygons(polygons,label,linewidth,color='red'):
    
    for poly in polygons[:-1]:
        plt.gca().add_patch(Polygon(poly, fill=False, color=color,linewidth=linewidth, alpha=0.5))

    plt.gca().add_patch(Polygon(polygons[-1], fill=False, color=color,linewidth=linewidth, alpha=0.5,label=label))



choicenode = 10
datasetname = 'three'
dir_path = os.path.dirname(os.path.realpath(__file__))
three = "/datasetnfp/threenfp.json"
datasetpath = dir_path+'\\resultsGPU2\\'+datasetname+'\\'
infpfile = datasetpath+'nfp-infp '+datasetname+'.json'
graphpath = datasetpath+'/'+'graph '+datasetname+'.csv'
pointcoordfile = datasetpath+'pointCoordinate '+datasetname+'.txt'
layermapfile = datasetpath+'LayerPoly'+datasetname+'.txt'
with open(infpfile) as file:
    polygons = json.load(file)



polygonYoffset = 9
layerOffset = 25
pos = {}
posfullOriginal = {}
G = nx.Graph()

# Read line by line
print("reading point coordinate..")
with open(pointcoordfile, 'r') as file:
    for line in file:
        if line[0] =='#':
            continue
        else:
            line = line.strip()
            s = line.split(' ')
            posfullOriginal[int(s[3])]={"x":float(s[1]),"y":float(s[2]),"Layer":int(s[0])}
            pos[int(s[3])] = (float(s[1])+((int(s[0])-1)*layerOffset), #x*offsetofLayer
                              float(s[2])) #y
            G.add_node(int(s[3]))

layermap = {}
with open(layermapfile, 'r') as file:
    lines = file.readlines()
print("reading layer map...")
for line in lines:
    if line[0] =='#':
            continue
    else:
        line = line.strip()
        s = line.split('\t')
        layermap[int(s[0])] = s[1]


print(posfullOriginal[choicenode]["Layer"])
print(layermap[posfullOriginal[choicenode]["Layer"]])
piece = layermap[posfullOriginal[choicenode]["Layer"]]

bins = copy.deepcopy(polygons['rect'])
del polygons['rect']

binpieces = []
for key,value in layermap.items():
    binlayer = []
    for v in bins:
        binlayer.append((v['x']+((int(key)-1)*layerOffset),v['y']))
    binpieces.append(binlayer)


refpoint = polygons[piece]['VERTICES'][0]

nfpDraw = []
mappedNFPS = {}
for poly in polygons[piece]['nfps']:
    mappedNFPS[poly['POLYGON']] = poly['VERTICES']

for key,value in layermap.items():
    nfptup = []
    for v in mappedNFPS[value]:
        vx = v['x']-refpoint['x']+posfullOriginal[choicenode]['x']+((int(key)-1)*layerOffset)
        vy = v['y']-refpoint['y']+posfullOriginal[choicenode]['y']
        nfptup.append((vx,vy))
    nfpDraw.append(nfptup)

#for nfpoly in range(len(polygons[piece]['nfps'])):
#    nfptup=[]
#    for v in polygons[piece]['nfps'][nfpoly]['VERTICES']:
#        v['x'] = v['x']-refpoint['x']+posfullOriginal[choicenode]['x']+(nfpoly*layerOffset)
#        v['y'] = v['y']-refpoint['y']+posfullOriginal[choicenode]['y']
#        nfptup.append((v['x'],v['y']))
#    nfpDraw.append(nfptup)



innerfits = []
for key,value in layermap.items():
    innerlayer = []
    for v in polygons[value]['innerfit']:
        innerlayer.append((v['x']+((int(key)-1)*layerOffset),v['y']))
    innerfits.append(innerlayer)

polygonsdraws = []
for key,value in layermap.items():
    polylayer = []
    for v in polygons[value]['VERTICES']:
        polylayer.append((v['x']+((int(key)-1)*layerOffset),v['y']-polygonYoffset))
    polygonsdraws.append(polylayer)

print("loading edges dataset..")

with open(graphpath, 'r') as file:
    lines = file.readlines()
print("reading edge dataset...")
for line in lines:
    if line[0] =='#':
            continue
    else:
        line = line.strip()
        s = line.split('\t')
        G.add_edge(int(s[0]),int(s[1]))
print("drawing graph..")
drawGraph(G,pos,choicenode,True)
drawPolygons(nfpDraw,'NFP',1,'red')
drawPolygons(binpieces,'board',2,'green')
drawPolygons(innerfits,'innerfit of board',3,'orange')
drawPolygons(polygonsdraws,'polygons',2,'purple')
plt.legend()
plt.axis('equal')
plt.show()
#import code
#code.interact(local=locals())

