import subprocess
import json
import os
import zlib
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

# Example usage:
dir_path = os.path.dirname(os.path.realpath(__file__))
blaz = "/blaz.json"
with open(dir_path+blaz) as file:
        inputdataset = json.load(file)
height = 20
width = 30
result = run_js_script_function('./scriptpy.js', 'calcNFP_INFP', [inputdataset, height,width])
json_object = json.dumps(result, indent=2)
with open("blaznfp.json", "w") as outfile:
    outfile.write(json_object)
