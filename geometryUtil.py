import math
import copy
TOL = math.pow(10, -9)
def searchStartPoint(A, B, inside, NFP=None):
    print("start point triggered!")
    # searches for an arrangement of A and B such that they do not overlap
    # if an NFP is given, only search for startpoints that have not already been traversed in the given NFP
    # clone arrays
    A = copy.deepcopy(A)
    B = copy.deepcopy(B)

    # close the loop for polygons
    if A[0] != A[A['len']-1]:
        A[A['len']]=A[0]
        A['len'] +=1

    if B[0] != B[B['len']-1]:
        B[B['len']]=B[0]
        B['len'] += 1

    for i in range(A['len'] - 1):
        if not A[i].get('marked'):
            A[i]['marked'] = True
            for j in range(B['len']):
                B['offestx'] = A[i]['x'] - B[j]['x']
                B['offsety'] = A[i]['y'] - B[j]['y']

                B_inside = None
                for k in range(B['len']):
                    in_poly = point_in_polygon({'x': B[k]['x'] + B['offestx'], 'y': B[k]['y'] + B['offsety']}, A)
                    if in_poly is not None:
                        B_inside = in_poly
                        break
                
                if B_inside is None:  # A and B are the same
                    print("A and B are the same")
                    return None

                start_point = {'x': B['offestx'], 'y': B['offsety']}
                if ((B_inside and inside) or (not B_inside and not inside)) and not intersect(A, B) and not in_nfp(start_point, NFP):
                    print("new starting point! case A:")
                    print("boolean- in poly:",in_poly)
                    print("boolean- B_inside:",B_inside)
                    print(start_point)
                    return start_point

                # slide B along vector
                vx = A[i+1]['x'] - A[i]['x']
                vy = A[i+1]['y'] - A[i]['y']

                d1 = polygon_projection_distance(A, B, {'x': vx, 'y': vy})
                d2 = polygon_projection_distance(B, A, {'x': -vx, 'y': -vy})

                d = None

                # todo: clean this up
                if d1 is None and d2 is None:
                    pass
                elif d1 is None:
                    d = d2
                elif d2 is None:
                    d = d1
                else:
                    d = min(d1, d2)

                # only slide until no longer negative
                # todo: clean this up
                if d is not None and not almost_equal(d, 0) and d > 0:
                    pass
                else:
                    continue
                
                vd2 = vx*vx + vy*vy

                if d*d < vd2 and not almost_equal(d*d, vd2):
                    vd = math.sqrt(vx*vx + vy*vy)
                    vx *= d/vd
                    vy *= d/vd

                B['offestx'] += vx
                B['offsety'] += vy

                for k in range(B['len']):
                    in_poly = point_in_polygon({'x': B[k]['x'] + B['offestx'], 'y': B[k]['y'] + B['offsety']}, A)
                    if in_poly is not None:
                        B_inside = in_poly
                        break
                    
                start_point = {'x': B['offestx'], 'y': B['offsety']}
                if ((B_inside and inside) or (not B_inside and not inside)) and not intersect(A, B) and not in_nfp(start_point, NFP):
                    print("new starting point case B")
                    print(start_point)

                    return start_point
    print("None starting point!")
    return None

def point_in_polygon(point, polygon):
    
    # return True if point is in the polygon, False if outside, and None if exactly on a point or edge
    if not polygon or polygon['len'] < 3:
        return None
    
    inside = False
    offsetx = getattr(polygon, 'offsetx', 0)
    offsety = getattr(polygon, 'offsety', 0)
    
    j = polygon['len'] - 1
    for i in range(polygon['len']):
        xi = polygon[i]['x'] + offsetx
        yi = polygon[i]['y'] + offsety
        xj = polygon[j]['x'] + offsetx
        yj = polygon[j]['y'] + offsety
        
        if almost_equal(xi, point['x']) and almost_equal(yi, point['y']):
            return None  # no result
        
        if on_segment({'x': xi, 'y': yi}, {'x': xj, 'y': yj}, point):
            return None  # exactly on the segment
        
        if almost_equal(xi, xj) and almost_equal(yi, yj):  # ignore very small lines
            continue
        
        intersect = ((yi > point['y']) != (yj > point['y'])) and (point['x'] < (xj - xi) * (point['y'] - yi) / (yj - yi) + xi)
        if intersect:
            inside = not inside
        
        j = i
    #print("point in polygon check")
    #print(polygon)
    #print(point)
    #print(inside)
    return inside


def in_nfp(p, nfp):
    # returns true if point already exists in the given nfp
    if not nfp or len(nfp) == 0:
        return False
    for nfp_part in nfp:
        for point in nfp_part:
            if almost_equal(p['x'], point['x']) and almost_equal(p['y'], point['y']):
                return True
    return False

def almost_equal(a, b):
    return abs(a - b) < TOL

# returns true if p lies on the line segment defined by AB, but not at any endpoints
# may need work!

def on_segment(A, B, p):
    # vertical line
    if almost_equal(A['x'], B['x']) and almost_equal(p['x'], A['x']):
        if not almost_equal(p['y'], B['y']) and not almost_equal(p['y'], A['y']) and p['y'] < max(B['y'], A['y']) and p['y'] > min(B['y'], A['y']):
            return True
        else:
            return False

    # horizontal line
    if almost_equal(A['y'], B['y']) and almost_equal(p['y'], A['y']):
        if not almost_equal(p['x'], B['x']) and not almost_equal(p['x'], A['x']) and p['x'] < max(B['x'], A['x']) and p['x'] > min(B['x'], A['x']):
            return True
        else:
            return False

    # range check
    if (p['x'] < A['x'] and p['x'] < B['x']) or (p['x'] > A['x'] and p['x'] > B['x']) or (p['y'] < A['y'] and p['y'] < B['y']) or (p['y'] > A['y'] and p['y'] > B['y']):
        return False

    # exclude end points
    if (almost_equal(p['x'], A['x']) and almost_equal(p['y'], A['y'])) or (almost_equal(p['x'], B['x']) and almost_equal(p['y'], B['y'])):
        return False

    cross = (p['y'] - A['y']) * (B['x'] - A['x']) - (p['x'] - A['x']) * (B['y'] - A['y'])

    if abs(cross) > TOL:
        return False

    dot = (p['x'] - A['x']) * (B['x'] - A['x']) + (p['y'] - A['y']) * (B['y'] - A['y'])

    if dot < 0 or almost_equal(dot, 0):
        return False

    len2 = (B['x'] - A['x'])**2 + (B['y'] - A['y'])**2

    if dot > len2 or almost_equal(dot, len2):
        return False

    return True

# todo: swap this for a more efficient sweep-line implementation
# returnEdges: if set, return all edges on A that have intersections
def intersect(A, B):

    A_offsetx = getattr(A, 'offsetx', 0)
    A_offsety = getattr(A, 'offsety', 0)
    
    B_offsetx = getattr(B, 'offsetx', 0)
    B_offsety = getattr(B, 'offsety', 0)
    
    A = copy.deepcopy(A)
    B = copy.deepcopy(B)
    
    for i in range(A['len'] - 1):
        for j in range(B['len'] - 1):
            a1 = {'x': A[i]['x'] + A_offsetx, 'y': A[i]['y'] + A_offsety}
            a2 = {'x': A[i+1]['x'] + A_offsetx, 'y': A[i+1]['y'] + A_offsety}

            b1 = {'x': B[j]['x'] + B_offsetx, 'y': B[j]['y'] + B_offsety}
            b2 = {'x': B[j+1]['x'] + B_offsetx, 'y': B[j+1]['y'] + B_offsety}
            
            prev_b_index = B['len'] - 1 if j == 0 else j - 1
            prev_a_index = A['len'] - 1 if i == 0 else i - 1
            next_b_index = 0 if j + 1 == B['len'] - 1 else j + 2
            next_a_index = 0 if i + 1 == A['len'] - 1 else i + 2
            
            # go even further back if we happen to hit on a loop end point
            if B[prev_b_index] == B[j] or (almost_equal(B[prev_b_index]['x'], B[j]['x']) and almost_equal(B[prev_b_index]['y'], B[j]['y'])):
                prev_b_index = B['len'] - 1 if prev_b_index == 0 else prev_b_index - 1
            
            if A[prev_a_index] == A[i] or (almost_equal(A[prev_a_index]['x'], A[i]['x']) and almost_equal(A[prev_a_index]['y'], A[i]['y'])):
                prev_a_index = A['len'] - 1 if prev_a_index == 0 else prev_a_index - 1
            
            # go even further forward if we happen to hit on a loop end point
            if B[next_b_index] == B[j+1] or (almost_equal(B[next_b_index]['x'], B[j+1]['x']) and almost_equal(B[next_b_index]['y'], B[j+1]['y'])):
                next_b_index = 0 if next_b_index == B['len'] - 1 else next_b_index + 1
            
            if A[next_a_index] == A[i+1] or (almost_equal(A[next_a_index]['x'], A[i+1]['x']) and almost_equal(A[next_a_index]['y'], A[i+1]['y'])):
                next_a_index = 0 if next_a_index == A['len'] - 1 else next_a_index + 1
            
            a0 = {'x': A[prev_a_index]['x'] + A_offsetx, 'y': A[prev_a_index]['y'] + A_offsety}
            b0 = {'x': B[prev_b_index]['x'] + B_offsetx, 'y': B[prev_b_index]['y'] + B_offsety}
            
            a3 = {'x': A[next_a_index]['x'] + A_offsetx, 'y': A[next_a_index]['y'] + A_offsety}
            b3 = {'x': B[next_b_index]['x'] + B_offsetx, 'y': B[next_b_index]['y'] + B_offsety}
            
            if on_segment(a1, a2, b1) or (almost_equal(a1['x'], b1['x']) and almost_equal(a1['y'], b1['y'])):
                # if a point is on a segment, it could intersect or it could not. Check via the neighboring points
                b0in = point_in_polygon(b0, A)
                b2in = point_in_polygon(b2, A)
                #xor
                if (b0in is True and b2in is False) or (b0in is False and b2in is True):
                    return True
                else:
                    continue
            
            if on_segment(a1, a2, b2) or (almost_equal(a2['x'], b2['x']) and almost_equal(a2['y'], b2['y'])):
                # if a point is on a segment, it could intersect or it could not. Check via the neighboring points
                b1in = point_in_polygon(b1, A)
                b3in = point_in_polygon(b3, A)
                #xor
                if (b1in is True and b3in is False) or (b1in is False and b3in is True):
                    return True
                else:
                    continue
            
            if on_segment(b1, b2, a1) or (almost_equal(a1['x'], b2['x']) and almost_equal(a1['y'], b2['y'])):
                # if a point is on a segment, it could intersect or it could not. Check via the neighboring points
                a0in = point_in_polygon(a0, B)
                a2in = point_in_polygon(a2, B)
                #xor
                if (a0in is True and a2in is False) or (a0in is False and a2in is True):
                    return True
                else:
                    continue
            
            if on_segment(b1, b2, a2) or (almost_equal(a2['x'], b1['x']) and almost_equal(a2['y'], b1['y'])):
                # if a point is on a segment, it could intersect or it could not. Check via the neighboring points
                a1in = point_in_polygon(a1, B)
                a3in = point_in_polygon(a3, B)
                #xor
                if (a1in is True and a3in is False) or (a1in is False and a3in is True):
                    return True
                else:
                    continue
            
            p = line_intersect(b1, b2, a1, a2)
            if p is not None:
                return True
    
    return False

def line_intersect(A, B, E, F, infinite=False):
    a1 = B['y'] - A['y']
    b1 = A['x'] - B['x']
    c1 = B['x'] * A['y'] - A['x'] * B['y']
    a2 = F['y'] - E['y']
    b2 = E['x'] - F['x']
    c2 = F['x'] * E['y'] - E['x'] * F['y']
    
    denom = a1 * b2 - a2 * b1

    try:
        x = (b1 * c2 - b2 * c1) / denom
    except(ZeroDivisionError):
        x = math.inf

    try:
        y = (a2 * c1 - a1 * c2) / denom
    except(ZeroDivisionError):
        y = math.inf

    if not math.isfinite(x) or not math.isfinite(y):
        return None
    
    # lines are colinear
    # Commented out in original JS, keeping it commented here
    """
    cross_ABE = (E['y'] - A['y']) * (B['x'] - A['x']) - (E['x'] - A['x']) * (B['y'] - A['y'])
    cross_ABF = (F['y'] - A['y']) * (B['x'] - A['x']) - (F['x'] - A['x']) * (B['y'] - A['y'])
    if almost_equal(cross_ABE, 0) and almost_equal(cross_ABF, 0):
        return None
    """
    
    if not infinite:
        # coincident points do not count as intersecting
        if (abs(A['x'] - B['x']) >TOL and
            ((A['x'] < B['x'] and (x < A['x'] or x > B['x'])) or
             (A['x'] >= B['x'] and (x > A['x'] or x < B['x'])))):
            return None
        if (abs(A['y'] - B['y']) >TOL and
            ((A['y'] < B['y'] and (y < A['y'] or y > B['y'])) or
             (A['y'] >= B['y'] and (y > A['y'] or y < B['y'])))):
            return None
        if (abs(E['x'] - F['x']) >TOL and
            ((E['x'] < F['x'] and (x < E['x'] or x > F['x'])) or
             (E['x'] >= F['x'] and (x > E['x'] or x < F['x'])))):
            return None
        if (abs(E['y'] - F['y']) >TOL and
            ((E['y'] < F['y'] and (y < E['y'] or y > F['y'])) or
             (E['y'] >= F['y'] and (y > E['y'] or y < F['y'])))):
            return None
    
    return {'x': x, 'y': y}

# project each point of B onto A in the given direction, and return the distance
def polygon_projection_distance(A, B, direction):
    B_offsetx = getattr(B, 'offsetx', 0)
    B_offsety = getattr(B, 'offsety', 0)
    
    A_offsetx = getattr(A, 'offsetx', 0)
    A_offsety = getattr(A, 'offsety', 0)
    print("projections offsets")
    print(B_offsetx)
    print(B_offsety)
    print(A_offsetx)
    print(A_offsety)
    A = A.copy()
    B = B.copy()
    
    # close the loop for polygons
    if A[0] != A[A['len']-1]:
        A[A['len']]=A[0]
        A['len'] += 1
    
    if B[0] != B[B['len']-1]:
        B[B['len']]=B[0]
        B['len'] += 1
    
    edge_A = A
    edge_B = B
    
    distance = None
    
    for i in range(edge_B['len']):
        # the shortest/most negative projection of B onto A
        min_projection = None
        for j in range(edge_A['len'] - 1):
            p = {
                'x': edge_B[i]['x'] + B_offsetx,
                'y': edge_B[i]['y'] + B_offsety
            }
            s1 = {
                'x': edge_A[j]['x'] + A_offsetx,
                'y': edge_A[j]['y'] + A_offsety
            }
            s2 = {
                'x': edge_A[j+1]['x'] + A_offsetx,
                'y': edge_A[j+1]['y'] + A_offsety
            }
            
            if abs((s2['y'] - s1['y']) * direction['x'] - (s2['x'] - s1['x']) * direction['y']) < TOL:
                continue
            
            # project point, ignore edge boundaries
            d = point_distance(p, s1, s2, direction)
            if d is not None and (min_projection is None or d < min_projection):
                min_projection = d
        
        if min_projection is not None and (distance is None or min_projection > distance):
            distance = min_projection
    
    return distance

def point_distance(p, s1, s2, normal, infinite=False):
    normal = normalize_vector(normal)
    
    dir = {
        'x': normal['y'],
        'y': -normal['x']
    }
    
    pdot = p['x'] * dir['x'] + p['y'] * dir['y']
    s1dot = s1['x'] * dir['x'] + s1['y'] * dir['y']
    s2dot = s2['x'] * dir['x'] + s2['y'] * dir['y']
    
    pdotnorm = p['x'] * normal['x'] + p['y'] * normal['y']
    s1dotnorm = s1['x'] * normal['x'] + s1['y'] * normal['y']
    s2dotnorm = s2['x'] * normal['x'] + s2['y'] * normal['y']
    
    if not infinite:
        if ((pdot < s1dot or almost_equal(pdot, s1dot)) and (pdot < s2dot or almost_equal(pdot, s2dot))) or \
           ((pdot > s1dot or almost_equal(pdot, s1dot)) and (pdot > s2dot or almost_equal(pdot, s2dot))):
            return None  # dot doesn't collide with segment, or lies directly on the vertex
        
        if (almost_equal(pdot, s1dot) and almost_equal(pdot, s2dot) and pdotnorm > s1dotnorm and pdotnorm > s2dotnorm):
            return min(pdotnorm - s1dotnorm, pdotnorm - s2dotnorm)
        
        if (almost_equal(pdot, s1dot) and almost_equal(pdot, s2dot) and pdotnorm < s1dotnorm and pdotnorm < s2dotnorm):
            return -min(s1dotnorm - pdotnorm, s2dotnorm - pdotnorm)
    
    return -(pdotnorm - s1dotnorm + (s1dotnorm - s2dotnorm) * (s1dot - pdot) / (s1dot - s2dot))

def normalize_vector(vector):
    if almost_equal(vector['x']**2+vector['y']**2,1): return vector #already a unit vector
    magnitude = math.sqrt(vector['x']**2 + vector['y']**2)
    return {
        'x': vector['x'] / magnitude,
        'y': vector['y'] / magnitude
    }

def polygonSlideDistance(A, B, direction, ignore_negative=False):
        # Get or set default offsets
        A_offsetx = getattr(A, 'offsetx', 0)
        A_offsety = getattr(A, 'offsety', 0)
        
        B_offsetx = getattr(B, 'offsetx', 0)
        B_offsety = getattr(B, 'offsety', 0)
        
        # Create copies of input polygons
        A = copy.deepcopy(A)
        B = copy.deepcopy(B)
        
        # Close the loop for polygons
        if A[0] != A[A['len']-1]:
            A[A['len']]=A[0] #append for dictionary
            A['len'] +=1
        
        if B[0] != B[B['len']-1]:
            B[B['len']]=B[0] #append for dictionary
            B['len'] += 1
            
        
        edge_A = A
        edge_B = B
        distance = None
        
        # Normalize direction vector
        dir = normalize_vector(direction)
        
        # Calculate normal and reverse vectors
        #never utilised?
        normal = {
            'x': dir['y'],
            'y': -dir['x']
        }
        #never utilised?
        reverse = {
            'x': -dir['x'],
            'y': -dir['y']
        }
        
        # Main calculation loop
        for i in range(edge_B['len'] - 1):
            min_d = None
            for j in range(edge_A['len'] - 1):
                A1 = {
                    'x': edge_A[j]['x'] + A_offsetx,
                    'y': edge_A[j]['y'] + A_offsety
                }
                A2 = {
                    'x': edge_A[j+1]['x'] + A_offsetx,
                    'y': edge_A[j+1]['y'] + A_offsety
                }
                B1 = {
                    'x': edge_B[i]['x'] + B_offsetx,
                    'y': edge_B[i]['y'] + B_offsety
                }
                B2 = {
                    'x': edge_B[i+1]['x'] + B_offsetx,
                    'y': edge_B[i+1]['y'] + B_offsety
                }
                
                # Ignore extremely small lines
                if ((almost_equal(A1['x'], A2['x']) and almost_equal(A1['y'], A2['y'])) or
                   (almost_equal(B1['x'], B2['x']) and almost_equal(B1['y'], B2['y']))):
                    continue
                
                d = segment_distance(A1, A2, B1, B2, dir)
                
                if (d is not None and (distance is None or d < distance)):
                    if not ignore_negative or d > 0 or almost_equal(d, 0):
                        distance = d
        
        return distance

def segment_distance(A, B, E, F, direction):
    normal = {
        'x': direction['y'],
        'y': -direction['x']
    }
    
    reverse = {
        'x': -direction['x'],
        'y': -direction['y']
    }
    
    # Calculate dot products
    dot_A = A['x'] * normal['x'] + A['y'] * normal['y']
    dot_B = B['x'] * normal['x'] + B['y'] * normal['y']
    dot_E = E['x'] * normal['x'] + E['y'] * normal['y']
    dot_F = F['x'] * normal['x'] + F['y'] * normal['y']
    
    # Calculate cross products
    cross_A = A['x'] * direction['x'] + A['y'] * direction['y']
    cross_B = B['x'] * direction['x'] + B['y'] * direction['y']
    cross_E = E['x'] * direction['x'] + E['y'] * direction['y']
    cross_F = F['x'] * direction['x'] + F['y'] * direction['y']
    
    #never utilised?
    cross_AB_min = min(cross_A, cross_B)
    cross_AB_max = max(cross_A, cross_B)
    
    #never utilised?
    cross_EF_max = max(cross_E, cross_F)
    cross_EF_min = min(cross_E, cross_F)
    
    AB_min = min(dot_A, dot_B)
    AB_max = max(dot_A, dot_B)
    
    EF_max = max(dot_E, dot_F)
    EF_min = min(dot_E, dot_F)
    
    # Check for segments that will merely touch at one point
    if (almost_equal(AB_max, EF_min) or 
        almost_equal(AB_min, EF_max)):
        return None
        
    # Check for segments that miss each other completely
    if AB_max < EF_min or AB_min > EF_max:
        return None
    
    # Calculate overlap
    if ((AB_max > EF_max and AB_min < EF_min) or 
        (EF_max > AB_max and EF_min < AB_min)):
        overlap = 1
    else:
        min_max = min(AB_max, EF_max)
        max_min = max(AB_min, EF_min)
        max_max = max(AB_max, EF_max)
        min_min = min(AB_min, EF_min)
        overlap = (min_max - max_min) / (max_max - min_min)
    
    # Calculate cross products for colinearity check
    cross_ABE = (E['y'] - A['y']) * (B['x'] - A['x']) - (E['x'] - A['x']) * (B['y'] - A['y'])
    cross_ABF = (F['y'] - A['y']) * (B['x'] - A['x']) - (F['x'] - A['x']) * (B['y'] - A['y'])
    
    # Check if lines are colinear
    if almost_equal(cross_ABE, 0) and almost_equal(cross_ABF, 0):
        AB_norm = {
            'x': B['y'] - A['y'],
            'y': A['x'] - B['x']
        }
        EF_norm = {
            'x': F['y'] - E['y'],
            'y': E['x'] - F['x']
        }
        
        AB_norm_length = math.sqrt(AB_norm['x']**2 + AB_norm['y']**2)
        AB_norm['x'] /= AB_norm_length
        AB_norm['y'] /= AB_norm_length
        
        EF_norm_length = math.sqrt(EF_norm['x']**2 + EF_norm['y']**2)
        EF_norm['x'] /= EF_norm_length
        EF_norm['y'] /= EF_norm_length
        
        # Check if segment normals point in opposite directions
        # Segment normals must point in opposite directions
        if (abs(AB_norm['y'] * EF_norm['x'] - AB_norm['x'] * EF_norm['y']) < TOL and 
            AB_norm['y'] * EF_norm['y'] + AB_norm['x'] * EF_norm['x'] < 0):
            # Check if normal of AB segment points in same direction as given direction vector
            # Normal of AB segment must point in same direction as given direction vector
            norm_dot = AB_norm['y'] * direction['y'] + AB_norm['x'] * direction['x']
            if almost_equal(norm_dot, 0, TOL):
                return None
            if norm_dot < 0:
                return 0
        return None
    
    distances = []
    
    # Check coincident points and calculate distances
    if almost_equal(dot_A, dot_E):
        distances.append(cross_A - cross_E)
    elif almost_equal(dot_A, dot_F):
        distances.append(cross_A - cross_F)
    elif dot_A > EF_min and dot_A < EF_max:
        d = point_distance(A, E, F, reverse)
        if d is not None and almost_equal(d, 0):
            dB = point_distance(B, E, F, reverse, True)
            if dB < 0 or almost_equal(dB * overlap, 0):
                d = None
        if d is not None:
            distances.append(d)
    
    # Similar checks for point B
    if almost_equal(dot_B, dot_E):
        distances.append(cross_B - cross_E)
    elif almost_equal(dot_B, dot_F):
        distances.append(cross_B - cross_F)
    elif dot_B > EF_min and dot_B < EF_max:
        d = point_distance(B, E, F, reverse)
        if d is not None and almost_equal(d, 0):
            dA = point_distance(A, E, F, reverse, True)
            if dA < 0 or almost_equal(dA * overlap, 0):
                d = None
        if d is not None:
            distances.append(d)
    
    # Check points E and F against segment AB
    if dot_E > AB_min and dot_E < AB_max:
        d = point_distance(E, A, B, direction)
        if d is not None and almost_equal(d, 0):
            dF = point_distance(F, A, B, direction, True)
            if dF < 0 or almost_equal(dF * overlap, 0):
                d = None
        if d is not None:
            distances.append(d)
    
    if dot_F > AB_min and dot_F < AB_max:
        d = point_distance(F, A, B, direction)
        if d is not None and almost_equal(d, 0):
            dE = point_distance(E, A, B, direction, True)
            if dE < 0 or almost_equal(dE * overlap, 0):
                d = None
        if d is not None:
            distances.append(d)
    
    if not distances:
        return None
    
    return min(distances)