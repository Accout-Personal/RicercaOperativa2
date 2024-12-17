import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os,json
def preparePlot():
    
    plt.figure(figsize=(16, 12))
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Nodes with Coordinate Axes')


def drawPolygons(polygons,label,linewidth,color='red'):
    
    for poly in polygons[:-1]:
        plt.gca().add_patch(Polygon(poly, fill=False, color=color,linewidth=linewidth, alpha=0.5))

    plt.gca().add_patch(Polygon(polygons[-1], fill=False, color=color,linewidth=linewidth, alpha=0.5,label=label))


jakobs1 = "/datasetJson/json/jakobs2.json"
dir_path = os.path.dirname(os.path.realpath(__file__))
path = dir_path+jakobs1
with open(path) as file:
        inputdataset = json.load(file)

offset = 20
#row = 5
column = 5
polyAll = []
piece = []
r=0
c=0
annotationpos = []
for key,value in inputdataset.items():
    polylist = []
    piece.append(key)
    for v in value['VERTICES']:
        polylist.append((v['x']+(c*offset),v['y']+(r*offset)))
    annotationpos.append((5+(c*offset),-3+(r*offset),key))
    c +=1
    if c==column:
         c=0
         r+=1    
    polyAll.append(polylist)

preparePlot()
drawPolygons(polyAll,'jakobs1',3,color='blue')
for a in annotationpos:
    plt.annotate(a[2], 
               xy=(a[0],a[1]),
               bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5)
               )

plt.axis('equal')  
plt.autoscale()    
plt.show()