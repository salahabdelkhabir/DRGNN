import os
# from model_loader_static import ModelLoader
SERVER_ROOT = os.path.dirname(os.path.abspath(__file__))

class Config(object):
    FRONT_ROOT = os.path.join(SERVER_ROOT, 'build')
    DATA_FOLDER = os.path.join(SERVER_ROOT, 'txgnn_data_v2')
    STATIC_FOLDER = os.path.join(SERVER_ROOT, 'build/static')
    GNN = 'txgnn_v2'
    USE_NEO4J = False  # Set to False to use file-based implementation
    
    def __init__(self):
        # Validate critical paths on initialization
        self.validate_paths()
        
    def validate_paths(self):
        """Validate that critical paths exist"""
        # Check if DATA_FOLDER exists
        if not os.path.exists(self.DATA_FOLDER):
            print(f"WARNING: Data folder not found at {self.DATA_FOLDER}")
            
            # Try to find the data folder in parent directories
            parent_dir = os.path.dirname(SERVER_ROOT)
            alternate_path = os.path.join(parent_dir, 'txgnn_data_v2')
            if os.path.exists(alternate_path):
                print(f"Found data folder at alternate location: {alternate_path}")
                self.DATA_FOLDER = alternate_path
            else:
                print("ERROR: Could not find txgnn_data_v2 folder. Application will not function correctly.")
        
        # Check for critical data files
        critical_files = [
            "filtered_predictions.csv",
            "drug_indication_subset.pkl",
            "node_types.json",
            "edge_types.json",
            "node_name_dict.json",
            "disease_options.json"
        ]
        
        missing_files = []
        for file in critical_files:
            file_path = os.path.join(self.DATA_FOLDER, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
        
        if missing_files:
            print(f"WARNING: Missing critical data files: {', '.join(missing_files)}")
            print("The application may not function correctly without these files.")

class ProductionConfig(Config):
    pass


class DevelopmentConfig(Config):
    pass


class TestingConfig(Config):
    pass
