import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

try:
    print("Reading notebook...")
    nb = nbformat.read('f5_tts_voice_cloning.ipynb', as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    print("Executing notebook...")
    ep.preprocess(nb, {'metadata': {'path': './'}})
    
    print("Notebook executed successfully.")
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            print(f"\n--- Output of Cell {i+1} ---")
            for output in cell.outputs:
                if output.output_type == 'stream':
                    print(output.text.strip())
                elif output.output_type == 'error':
                    print(f"ERROR: {output.ename}: {output.evalue}")
                    
except Exception as e:
    print(f"Execution failed: {e}")
