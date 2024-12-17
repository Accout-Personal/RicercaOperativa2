
    import './geometryutil.js';
    import './clipper.js';
    var useHoles = false //no holes for current project
    export function calcNFP_INFP(inputdataset,height,width) {
        //const fileInput = document.getElementById('jsonFileInput');
        //const file = fileInput.files[0];
        //const height = document.getElementById('height').value;
        //const width = document.getElementById('width').value;
        if (height <= 0 || height == ""){
            alert("invalid height value")
            return;
        }        

        if (width <= 0 || width == ""){
            alert("invalid width value")
            return;
        }

        const rect = [{"x":0,"y":0},{"x":Number(height),"y":0},{"x":Number(height),"y":Number(width)},{"x":0,"y":Number(width)}]
        
        try {
            var dataset = inputdataset;
            console.log('Loaded JSON data:', dataset);
            //iterate through all polygons
            for (let [fixedPolyKey,fixedPolyValue] of Object.entries(dataset)) {
                console.log("poly key is: ");
                console.log(fixedPolyKey);
                
                //generate the innerfit polygon for rectangle
                console.log
                console.log("calculating Inner-fit polygon of rectangle for",fixedPolyKey)
                var NFP = svglite(rect,dataset[fixedPolyKey].VERTICES,true,false)
                dataset[fixedPolyKey].innerfit = NFP[0];
                dataset[fixedPolyKey].nfps = [];
                for (let [rotPolyKey,rotPolyValue] of Object.entries(dataset)){
                    console.log(rotPolyKey + " rotate around " + fixedPolyKey);
                    console.log(dataset[fixedPolyKey].VERTICES);
                    console.log(dataset[rotPolyKey].VERTICES);
                    var nfp = {"POLYGON":rotPolyKey};
                    nfp.VERTICES = svglite(dataset[fixedPolyKey].VERTICES,dataset[rotPolyKey].VERTICES,false,false)[0];
                    dataset[fixedPolyKey].nfps.push(nfp);
                }
              }
            dataset.rect = rect;
            return dataset;
            
            } catch (error) {
                console.error(error);
                alert(error.message);
            }
    };

    //given a static polygon A and a movable polygon B, compute a no fit polygon by orbiting B about A
    //if the inside flag is set, B is orbited inside of A rather than outside
    //if the searchEdges flag is set, all edges of A are explored for NFPs - multiple #
    //Input:
    // A,B:list of array{x,y}[](points?)
    // Input,searchEdges(boolean)
    function svglite(A,B,inside,searchEdges){
            
            var nfp;
            if(inside){
                if(GeometryUtil.isRectangle(A, 0.001)){
                    nfp = GeometryUtil.noFitPolygonRectangle(A,B);
                }
                else{
                    nfp = GeometryUtil.noFitPolygon(A,B,true,searchEdges);
                }
                
                // ensure all interior NFPs have the same winding direction
                if(nfp && nfp.length > 0){
                    for(var i=0; i<nfp.length; i++){
                        if(GeometryUtil.polygonArea(nfp[i]) > 0){
                            nfp[i].reverse();
                        }
                    }
                }
                else{
                    // warning on null inner NFP
                    // this is not an error, as the part may simply be larger than the bin or otherwise unplaceable due to geometry
                    log('NFP Warning: no inner NFP');
                }
            }
            else{
                if(searchEdges){
                    nfp = GeometryUtil.noFitPolygon(A,B,false,searchEdges);
                }
                else{
                    nfp = minkowskiDifference(A,B);
                }
                // sanity check
                if(!nfp || nfp.length == 0){
                    log('NFP Error !');
                    log('A: ',JSON.stringify(A));
                    log('B: ',JSON.stringify(B));
                    return null;
                }
                
                for(var i=0; i<nfp.length; i++){
                    if(!searchEdges || i==0){ // if searchedges is active, only the first NFP is guaranteed to pass sanity check
                        if(Math.abs(GeometryUtil.polygonArea(nfp[i])) < Math.abs(GeometryUtil.polygonArea(A))){
                            log('NFP Area Error: ', Math.abs(GeometryUtil.polygonArea(nfp[i])));
                            log('NFP:', JSON.stringify(nfp[i]));
                            log('A: ',JSON.stringify(A));
                            log('B: ',JSON.stringify(B));
                            nfp.splice(i,1);
                            return null;
                        }
                    }
                }
                
                if(nfp.length == 0){
                    return null;
                }
                
                // for outer NFPs, the first is guaranteed to be the largest. Any subsequent NFPs that lie inside the first are holes
                for(var i=0; i<nfp.length; i++){
                    if(GeometryUtil.polygonArea(nfp[i]) > 0){
                        nfp[i].reverse();
                    }
                    
                    if(i > 0){
                        if(GeometryUtil.pointInPolygon(nfp[i][0], nfp[0])){
                            if(GeometryUtil.polygonArea(nfp[i]) < 0){
                                nfp[i].reverse();
                            }
                        }
                    }
                }
                
                // generate nfps for children (holes of parts) if any exist
                if(useHoles && A.childNodes && A.childNodes.length > 0){
                    var Bbounds = GeometryUtil.getPolygonBounds(B);
                    
                    for(var i=0; i<A.childNodes.length; i++){
                        var Abounds = GeometryUtil.getPolygonBounds(A.childNodes[i]);

                        // no need to find nfp if B's bounding box is too big
                        if(Abounds.width > Bbounds.width && Abounds.height > Bbounds.height){
                        
                            var cnfp = GeometryUtil.noFitPolygon(A.childNodes[i],B,true,searchEdges);
                            // ensure all interior NFPs have the same winding direction
                            if(cnfp && cnfp.length > 0){
                                for(var j=0; j<cnfp.length; j++){
                                    if(GeometryUtil.polygonArea(cnfp[j]) < 0){
                                        cnfp[j].reverse();
                                    }
                                    nfp.push(cnfp[j]);
                                }
                            }
                        
                        }
                    }
                }
            }
            
            function log(){
                if(typeof console !== "undefined") {
                    console.log.apply(console,arguments);
                }
            }
            
            function toClipperCoordinates(polygon){
                var clone = [];
                for(var i=0; i<polygon.length; i++){
                    clone.push({
                        X: polygon[i].x,
                        Y: polygon[i].y
                    });
                }

                return clone;
            };
            
            function toNestCoordinates(polygon, scale){
                var clone = [];
                for(var i=0; i<polygon.length; i++){
                    clone.push({
                        x: polygon[i].X/scale,
                        y: polygon[i].Y/scale
                    });
                }

                return clone;
            };
            
            function minkowskiDifference(A, B){
                var Ac = toClipperCoordinates(A);
                ClipperLib.JS.ScaleUpPath(Ac, 10000000);
                var Bc = toClipperCoordinates(B);
                ClipperLib.JS.ScaleUpPath(Bc, 10000000);
                for(var i=0; i<Bc.length; i++){
                    Bc[i].X *= -1;
                    Bc[i].Y *= -1;
                }
                var solution = ClipperLib.Clipper.MinkowskiSum(Ac, Bc, true);
                var clipperNfp;
                
                var largestArea = null;
                console.log("solution array:",solution)
                for(i=0; i<solution.length; i++){
                    var n = toNestCoordinates(solution[i], 10000000);
                    var sarea = GeometryUtil.polygonArea(n);
                    if(largestArea === null || largestArea > sarea){
                        console.log("the n is:",n)
                        clipperNfp = n;
                        largestArea = sarea;
                    }
                }
                if (!clipperNfp){ return []}
                
                for(var i=0; i<clipperNfp.length; i++){
                    clipperNfp[i].x += B[0].x;
                    clipperNfp[i].y += B[0].y;
                }
                   
    
                return [clipperNfp];
            }
            return nfp;
    };
