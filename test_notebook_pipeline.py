import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import glob

def test_notebook(notebook_path: str):
    print(f"Testing notebook: {notebook_path}")
    
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
        
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    try:
        ep.preprocess(nb, {'metadata': {'path': './'}})
        print(f"✅ Notebook {notebook_path} executed successfully.")
    except Exception as e:
        print(f"❌ Notebook {notebook_path} failed to execute.")
        print(e)
        raise

if __name__ == "__main__":
    notebook_file = "f5_tts_voice_cloning.ipynb"
    test_notebook(notebook_file)
