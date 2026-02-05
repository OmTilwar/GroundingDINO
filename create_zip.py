import os
import zipfile

def zip_project(output_filename="GroundingDINO_Project.zip"):
    # Folders/Files to explicitly EXCLUDE
    # These match the requested lightweight rules
    EXCLUDE_DIRS = {
        'venv', 
        'weights', 
        'outputs', 
        '__pycache__', 
        '.git', 
        '.idea', 
        '.vscode', 
        'build', 
        'dist', 
        '.egg-info',
        'GroundingDINO.egg-info',
        'tmp'
    }
    
    EXCLUDE_EXTENSIONS = {
        '.pth', 
        '.pt', 
        '.onnx', 
        '.zip', 
        '.pyc', 
        '.pyd'
    }

    print(f"Zipping project to {output_filename}...")
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk mostly everything from current directory
        for root, dirs, files in os.walk('.'):
            # Modify dirs in-place to skip excluded directories
            # This prevents os.walk from entering them
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]
            
            for file in files:
                # Skip excluded extensions
                _, ext = os.path.splitext(file)
                if ext.lower() in EXCLUDE_EXTENSIONS:
                    continue
                
                # Check specific excluded filenames
                if file in ['gpu_log.txt', 'output_log.txt', output_filename]:
                    continue

                file_path = os.path.join(root, file)
                # Archive name matches relative path
                zipf.write(file_path, file_path)
                
    print("Done! Zip created successfully.")

if __name__ == "__main__":
    zip_project()
