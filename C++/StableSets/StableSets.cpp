#include <ilcplex/ilocplex.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <tuple>
#include <algorithm>
#include <cmath>

using namespace std;

// Fast CSV reading using memory mapping
vector<pair<unsigned int, unsigned int>> readEdgesFromCSV(const string& filename, int edge_size) {
    auto start = chrono::high_resolution_clock::now();

    vector<pair<unsigned int, unsigned int>> edges;
    edges.reserve(edge_size);  // Pre-allocate for expected size

    ifstream file(filename, ios::binary | ios::ate);
    if (!file) throw runtime_error("Cannot open file");

    // Get file size for memory mapping
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);

    // Read file into memory buffer
    vector<char> buffer(size);
    if (!file.read(buffer.data(), size))
        throw runtime_error("Cannot read file");

    // Process buffer
    stringstream ss(string(buffer.data(), size));
    string line;

    // Skip header if exists
    getline(ss, line);

    // Process lines
    while (getline(ss, line)) {
        size_t pos = line.find('\t');
        if (pos != string::npos) {
            int src = stoi(line.substr(0, pos));
            int dst = stoi(line.substr(pos + 1));
            edges.emplace_back(src, dst);
        }
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "CSV reading time: " << duration.count() << "ms\n";
    cout << "Number of edges read: " << edges.size() << "\n";
    if (edges.size() <= 0 ) throw runtime_error("No edge read!");
    return edges;
}


tuple<int,int> readMetadataFromCSV(const string& filename) {
    
    ifstream file(filename, ios::binary | ios::ate);
    if (!file) throw runtime_error("Cannot open file");

    // Get file size for memory mapping
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);

    // Read file into memory buffer
    vector<char> buffer(size);
    if (!file.read(buffer.data(), size))
        throw runtime_error("Cannot read file");

    // Process buffer
    stringstream ss(string(buffer.data(), size));
    string line;

    // Skip header if exists
    getline(ss, line);

    // Process lines
    int target_size = 0;
    int edge_size = 0;
    while (getline(ss, line)) {
        size_t pos = line.find('\t');
        if (pos != string::npos) {
            string firstPart = line.substr(0, pos);
            if (firstPart == "Total Pieces :") {
                target_size = stoi(line.substr(pos + 1));
            }
            else if (firstPart == "Intra Layer Edges:"){
                edge_size = stoi(line.substr(pos + 1));
            }
        }
    }

    if (target_size == 0 || edge_size ==0){ throw runtime_error("size error!"); }

    return make_tuple(target_size, edge_size);
}



tuple<vector<unsigned int>, vector<unsigned int>> readCliquesFromCSV(const string& filename) {

    ifstream file(filename, ios::binary | ios::ate);
    if (!file) throw runtime_error("Cannot open file");

    // Get file size for memory mapping
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);

    // Read file into memory buffer
    vector<char> buffer(size);
    if (!file.read(buffer.data(), size))
        throw runtime_error("Cannot read file");

    // Process buffer
    stringstream ss(string(buffer.data(), size));
    string line;

    // Skip header if exists
    getline(ss, line);

    // Process lines
    vector<unsigned int> clique_start = {};
    vector<unsigned int> clique_end = {};
    while (getline(ss, line)) {
        size_t pos = line.find('\t');
        if (pos != string::npos) {
            clique_start.push_back(stoi(line.substr(0, pos)));
            clique_end.push_back(stoi(line.substr(pos+1)));
        }
    }

    if (clique_start.size() == 0 || clique_end.size() == 0) { throw runtime_error("size error!"); }

    return { clique_start, clique_end };
}

vector<vector<int>> readCoords(const string& filename,int coordsize) {
    ifstream file(filename, ios::binary | ios::ate);
    if (!file) throw runtime_error("Cannot open file");

    // Get file size for memory mapping
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);

    // Read file into memory buffer
    vector<char> buffer(size);
    if (!file.read(buffer.data(), size))
        throw runtime_error("Cannot read file");

    // Process buffer
    stringstream ss(string(buffer.data(), size));
    string line;

    // Skip header if exists
    getline(ss, line);
    vector<vector<int>> data;
    data.reserve(coordsize);
    
    int layer, x, y, id;    
    while (ss >> layer >> x >> y >> id) {
        data.push_back({layer, x, y});
    }
    file.close();

    return data;
}

vector<unsigned int> findMaxLayerX(vector<vector<int>> offsetMap, vector<unsigned int> clique_starts, vector<unsigned int> clique_ends) {
    vector<unsigned int> MaxXLayer(clique_ends.size(),0);

    for (unsigned int i = 0; i < MaxXLayer.size(); i++) {
        for (unsigned int j = clique_starts[i]; j < clique_ends[i]; j++) {
            if (offsetMap[j][1] > MaxXLayer[i]) {
                MaxXLayer[i] = offsetMap[j][1];
            }
        }
        //MaxXLayer[i] = MaxXLayer[i];
    }
    return MaxXLayer;
}

vector<vector<unsigned int>> generateCliqueVector(vector<unsigned int> clique_start, vector<unsigned int> clique_end){
    vector<vector<unsigned  int>> cliqueVector(clique_start.size());
    for (unsigned int i = 0; i < clique_start.size(); i++) {
        size_t size = static_cast<size_t>(clique_end[i] - clique_start[i]) + 1;
        cliqueVector[i].resize(size);
        for (unsigned int j = 0; j <= clique_end[i]-clique_start[i]; j++) {
            cliqueVector[i][j] = clique_start[i] + j;
        }
    }
    return cliqueVector;
}

//this returns clique coords and edges filtered
tuple< vector<pair<unsigned int, unsigned int>>, vector<vector<unsigned int>> , vector<bool> > filterNodes(
    vector<vector<int>> coordMap, 
    vector<vector<unsigned int>> CliqueCoords, 
    vector<pair<unsigned int, unsigned int>> edges,
    vector<unsigned int> MaxXLayer, 
    vector<int> LastCutThreashold,
    unsigned int binLength, 
    unsigned int cutlength
)
{
    unsigned int layerLength = CliqueCoords.size();
    vector<vector<bool>> clique_remove(layerLength);
    for (int i = 0; i < layerLength; i++) {
        clique_remove[i] = vector<bool>(CliqueCoords[i].size(), false);
    }

    vector<int> LayerCutThreashold(layerLength);
    for (int i = 0; i < layerLength; i++) {
        int boardCutOffset = binLength - cutlength;
        LayerCutThreashold[i] = MaxXLayer[i] - boardCutOffset;
    }
    //TODO: implement adding element in case of current cut>last cut for memory optimization.
    
    vector<bool> edges_remove(edges.size(), false);
    // Mark elements for removal
    for (unsigned int i = 0; i < edges.size(); i++) {

        //int xa = coordMap[edges[i][0]][1]; //0:layer 1:x 2:y
        //int xb = coordMap[edges[i][1]][1];
        // 
        //unsigned int layera = coordMap[edges[i][0]][0];
        //unsigned int layerb = coordMap[edges[i][1]][0];
        // 
        //unsigned layerCuta = LayerCutThreashold[coordMap[edges[i][0]][0]]
        //unsigned layerCutb = LayerCutThreashold[coordMap[edges[i][1]][0]]

        // x of node a or b is bigger than cutThreashold of its layer then remove edge

        if (/*node a check: */coordMap[edges[i].first][1] > LayerCutThreashold[coordMap[edges[i].first][0]]  ||
            /*node b check: */coordMap[edges[i].second][1] > LayerCutThreashold[coordMap[edges[i].second][0]]) {
            edges_remove[i] = true;
        }
    }

    vector<pair<unsigned int, unsigned int>> Filtereddedges = edges;
    // Single pass removal
    auto write = Filtereddedges.begin();
    for (auto read = Filtereddedges.begin(); read != Filtereddedges.end(); ++read) {
        if (!edges_remove[read - Filtereddedges.begin()]) {
            if (write != read) {
                *write = std::move(*read);
            }
            ++write;
        }
    }
    Filtereddedges.erase(write, Filtereddedges.end());
    vector<bool>().swap(edges_remove);

    vector<vector<unsigned int>> FilteredCliqueCoords = CliqueCoords;
    // Mark elements for removal
    for (unsigned int i = 0; i < layerLength; i++) {
        for (unsigned int j = 0; j < FilteredCliqueCoords[i].size(); j++)
        {
            if (/*node check: */coordMap[FilteredCliqueCoords[i][j]][1] > LayerCutThreashold[coordMap[FilteredCliqueCoords[i][j]][0]]) {
                clique_remove[i][j] = true;
            }
        }  
    }

    for (unsigned int i = 0; i < layerLength; i++) {
        auto write = FilteredCliqueCoords[i].begin();
        for (auto read = FilteredCliqueCoords[i].begin(); read != FilteredCliqueCoords[i].end(); ++read) {
            if (!clique_remove[i][read - FilteredCliqueCoords[i].begin()]) {
                if (write != read) {
                    *write = std::move(*read);
                }
                ++write;
            }
        }
        FilteredCliqueCoords[i].erase(write, FilteredCliqueCoords[i].end());
    }
    cout << "N Edges after filter: " << Filtereddedges.size() << "\n";
    //flatten the booleanMap
    //Calculate total size needed
    
    size_t totalSize = 0;
    for (const auto& vec : clique_remove) {
        totalSize += vec.size();
    }

    // Create result vector and reserve space
    vector<bool> NodeRemoveMap;
    NodeRemoveMap.reserve(totalSize);

    // Concatenate all vectors
    for (const auto& vec : clique_remove) {
        NodeRemoveMap.insert(NodeRemoveMap.end(), vec.begin(), vec.end());
    }

    return {Filtereddedges, FilteredCliqueCoords,NodeRemoveMap};

}



void print2dVector(const vector<vector<int>> &vec) {
    
    for (const auto& row : vec) {
        for (int val : row) {
            std::cout << val << " \n";
        }
    }
    std::cout << std::endl;
}

void print2dVectorUnsigned(const vector<vector<unsigned int>>& vec) {

    for (const auto& row : vec) {
        for (int val : row) {
            std::cout << val << " \n";
        }
    }
    std::cout << std::endl;
}

int getLayerFromPoint(int point) {

}


int main(int argc, char* argv[]) {

    string datasetname = "shapes4";
    //if (argc != 3) {
   //     cerr << "Usage: " << argv[0] << " <csv_file> <target_size>\n";
   //     return 1;
   // }
    string masterDirectory = "C:/Users/Leo/Desktop/Gran Finale/ProgramMath/dataset/resultsGPU3/" + datasetname + "/";
    string filename = masterDirectory + "graph " + datasetname + ".csv";
    string metadata = masterDirectory + "metadata" + datasetname + ".csv";
    string cliquedata = masterDirectory + "cliques" + datasetname + ".csv";
    string coordsmap = masterDirectory + "pointCoordinate " + datasetname + ".txt";
    //TODO: Add function read lenght from metadata
    //TODO: Add function read total area from metadata
    unsigned int boardwidth = 40;
    unsigned int boardLenght = 28;
    unsigned int boardArea = boardLenght * boardwidth;

    unsigned int totalPieceArea = 640; //TODO da verificare
    unsigned int PieceAreaBoardLength = ceil(totalPieceArea / boardwidth); // 16

    IloEnv env;
    try {
        //Read metadata from CSV
        tuple<int, int>meta = readMetadataFromCSV(metadata);
        int target_size = get<0>(meta); //this has to be int, otherwise CPLEX would give an error.
        unsigned int edge_size = get<1>(meta);
        vector<unsigned int> clique_start;
        vector<unsigned int> clique_end;
        tie (clique_start, clique_end) = readCliquesFromCSV(cliquedata);
        int coordSize = (*max_element(clique_end.begin(), clique_end.end()))+1;
        cout <<"Number of Nodes: " << coordSize << "\n";
        // Read edges from CSV

        cout << "Reading Edges From CSV \n";
        auto edges = readEdgesFromCSV(filename, edge_size);
        cout << "Reading coordinate Map from CSV \n";
        vector<vector<int>> coordMap = readCoords(coordsmap, coordSize);
        cout << "Finding max X per Layer \n";
        vector<unsigned int> MaxXLayer = findMaxLayerX(coordMap,clique_start,clique_end);

        cout << "Generating Clique Vectors \n";
        vector<vector<unsigned int>> CliqueCoords = generateCliqueVector(clique_start,clique_end);
        //print2dVectorUnsigned(CliqueCoords);
        
        vector<int> LastCutThreashold(clique_start.size(), boardLenght); // TODO:implement this.
        vector<vector<unsigned int>> FilteredClique;
        vector<pair<unsigned int, unsigned int>> FilteredEdge;
        vector<bool> NodeRemoveMap;
        cout << "Filtering Elements...\n";
        auto start_filter = chrono::high_resolution_clock::now();
        tie(FilteredEdge, FilteredClique, NodeRemoveMap) = filterNodes(coordMap, CliqueCoords, edges, MaxXLayer, LastCutThreashold, boardLenght, PieceAreaBoardLength);
        //TODO: Apply filter by using the coordmap->find layer and coord->apply offset check
        auto end_filter = chrono::high_resolution_clock::now();
        auto duration_filter = chrono::duration_cast<chrono::milliseconds>(end_filter - start_filter);
        cout << "Filtering time: " << duration_filter.count() << "ms\n";
        //TODO: Filter coordinates given the Length.
        // Find number of vertices
        unsigned int n_vertices = coordMap.size();

        // Create model
        IloModel model(env);

        // Create binary variables
        IloBoolVarArray vars(env, n_vertices);

        cout << "Adding constraint...\n";
        auto start = chrono::high_resolution_clock::now();


        cout << "Adding edge constraints...\n";
        // Add edge constraints
        for (const auto& edge : FilteredEdge) {
            model.add(vars[edge.first] + vars[edge.second] <= 1);
        }

        cout << "Adding clique constraints...\n";
        //Add Cliques
        for (int i = 0; i < clique_start.size(); ++i) {
            // Add constraint that only one node from clique can be selected
            IloExpr clique_sum(env);
            for (unsigned int node = 0; node < FilteredClique[i].size(); ++node) {
                clique_sum += vars[FilteredClique[i][node]];
            }
            model.add(clique_sum <= 1);
            clique_sum.end();
        }

        // Create objective: maximize sum of variables
        IloExpr obj(env);
        for (unsigned int i = 0; i < n_vertices; i++) {
            if (!NodeRemoveMap[i]) obj += vars[i];
        }
        model.add(IloMaximize(env,obj));

        target_size = 16;
        // Add target size constraint if specified
        if (target_size > 0) {
            IloExpr sum(env);
            for (int i = 0; i < n_vertices; i++) {
                if (!NodeRemoveMap[i]) sum += vars[i];
            }
            model.add(sum >= target_size);
            sum.end();
        }
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        cout << "Adding constraint time: " << duration.count() << "ms\n";

        cout << "Removing Variable ... \n";
        auto start_removevar = chrono::high_resolution_clock::now();

        //Remove the cut variables
        //Note: this will also remove from the objective function and cliques.
        //Note2: byt this is incredibly slow...
        //for (IloInt i = 0; i < n_vertices; i++) {
        //    if (NodeRemoveMap[i]) {
        //        model.remove(vars[i]);
        //    }
        //}
        auto end_removevar = chrono::high_resolution_clock::now();
        auto duration_removevar = chrono::duration_cast<chrono::milliseconds>(end_removevar - start_removevar);
        cout << "Removing variable time: " << duration_removevar.count() << "ms\n";

        cout << "Solving the problem...\n";
        // Create solver instance
        IloCplex cplex(model);
        cplex.setParam(IloCplex::Param::MIP::Tolerances::MIPGap, 1.0);
        cplex.setParam(IloCplex::Param::MIP::Limits::Solutions, 1);
        cplex.setParam(IloCplex::Param::MIP::Strategy::Probe, 3);
        cplex.setParam(IloCplex::Param::Threads, 14);

        start = chrono::high_resolution_clock::now();

        if (cplex.solve()) {
            end = chrono::high_resolution_clock::now();
            duration = chrono::duration_cast<chrono::milliseconds>(end - start);
            cout << "Solving time: " << duration.count() << "ms\n";

            // Get solution
            vector<int> stable_set;
            for (int i = 0; i < n_vertices; i++) {
                if (cplex.getValue(vars[i]) > 0.5) {
                    stable_set.push_back(i);
                }
            }

            cout << "Stable set size: " << stable_set.size() << "\n";
        }
        else {
            cout << "No solution found\n";
        }
    }
    catch (IloException& e) {
        cerr << "Error: " << e << endl;
    }
    env.end();
    return 0;
}