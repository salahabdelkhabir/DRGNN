import os
import json
import logging
import pandas as pd
import time  
import pickle 
from typing import List, Dict, Any
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from flask import current_app, g
from mykeys import get_keys
import pandas as pd
import networkx as nx
from collections import defaultdict



class FileBasedGraphDatabase:
    """A drop-in replacement for Neo4jApp that uses files instead of Neo4j"""
    
    def __init__(self, server="txgnn_v2", datapath='./txgnn_data_v2/', **kwargs):
        self.data_path = datapath
        self.node_types = [
            "anatomy",
            "biological_process",
            "cellular_component",
            "disease",
            "drug",
            "effect/phenotype",
            "exposure",
            "gene/protein",
            "molecular_function",
            "pathway"
        ]
        
        # Initialize the graph
        self.graph = nx.MultiDiGraph()
        self.node_types_dict = {}
        self.node_names = {}
        
        # Load all required data
        print("Starting data loading process...")
        start_time = time.time()
        
        # Load with optimized approach
        self.load_graph_data_optimized()
        self.load_predictions()
        self.load_drug_indications()
        
        end_time = time.time()
        print(f"Total data loading time: {end_time - start_time:.2f} seconds")
        
        # For compatibility with the Neo4j version
        self.session = None
    
    def create_session(self):
        """No-op for compatibility with Neo4jApp"""
        pass
    
    def close_session(self):
        """No-op for compatibility with Neo4jApp"""
        pass
    
    def load_graph_data_optimized(self):
        """Load graph structure with optimization strategies"""
        pickle_path = os.path.join(self.data_path, "graph_data.pkl")
        
        # Check if pickle file exists and load it
        if os.path.exists(pickle_path):
            print(f"Loading preprocessed graph data from {pickle_path}")
            try:
                start_time = time.time()
                with open(pickle_path, 'rb') as f:
                    data = pickle.load(f)
                    self.graph = data['graph']
                    self.node_types_dict = data['node_types_dict']
                    self.node_names = data['node_names']
                end_time = time.time()
                print(f"Loaded graph data in {end_time - start_time:.2f} seconds")
                print(f"Loaded {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
                return
            except Exception as e:
                print(f"Error loading preprocessed data: {e}")
                print("Falling back to CSV loading")
        
        # Fall back to loading from CSV using chunking
        self.load_graph_data()
        
        # After loading from CSV, save to pickle for next time
        try:
            print(f"Saving preprocessed graph data to {pickle_path}")
            with open(pickle_path, 'wb') as f:
                pickle.dump({
                    'graph': self.graph,
                    'node_types_dict': self.node_types_dict,
                    'node_names': self.node_names
                }, f, protocol=4)  # protocol 4 works with larger data
            print("Preprocessing complete")
        except Exception as e:
            print(f"Error saving preprocessed data: {e}")
    
    def load_graph_data(self):
        """Load graph structure from graphmask output file with chunking"""
        attention_path = os.path.join(self.data_path, "graphmask_output_indication.csv")
        print(f"Loading graph data from {attention_path}")
        
        if os.path.exists(attention_path):
            start_time = time.time()
            
            # Use pandas chunk reading
            chunk_size = 100000  # Adjust based on your system's memory
            total_rows = 0
            
            print("Loading in chunks...")
            for chunk_idx, chunk in enumerate(pd.read_csv(attention_path, 
                                                       dtype={'x_id': 'string', 'y_id': 'string'},
                                                       chunksize=chunk_size)):
                chunk_start = time.time()
                # Process this chunk
                for _, row in chunk.iterrows():
                    src_id, src_type = row['x_id'], row['x_type']
                    tgt_id, tgt_type = row['y_id'], row['y_type']
                    rel_type = row['relation']
                    
                    # Add nodes with their types
                    if src_id not in self.node_types_dict:
                        self.graph.add_node(src_id)
                        self.node_types_dict[src_id] = src_type
                        self.node_names[src_id] = row['x_name']
                        
                    if tgt_id not in self.node_types_dict:
                        self.graph.add_node(tgt_id)
                        self.node_types_dict[tgt_id] = tgt_type
                        self.node_names[tgt_id] = row['y_name']
                    
                    # Add edge with attributes
                    self.graph.add_edge(
                        src_id, tgt_id, 
                        type=rel_type,
                        layer1_att=row['layer1_att'],
                        layer2_att=row['layer2_att'],
                        edge_info=rel_type
                    )
                
                total_rows += len(chunk)
                chunk_end = time.time()
                print(f"Processed chunk {chunk_idx+1}, rows: {len(chunk)}, " 
                      f"total rows: {total_rows}, time: {chunk_end - chunk_start:.2f} seconds")
            
            end_time = time.time()
            print(f"Loaded {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges "
                 f"in {end_time - start_time:.2f} seconds")
        else:
            print(f"Warning: File {attention_path} not found.")
    
    def load_predictions(self):
        """Load drug predictions from CSV file"""
        predictions_path = os.path.join(self.data_path, "filtered_predictions.csv")
        print(f"Loading predictions from {predictions_path}")
        
        if os.path.exists(predictions_path):
            try:
                start_time = time.time()
                
                # Create dictionary to store predictions
                self.drug_predictions = {}
                
                import csv
                with open(predictions_path, 'r') as f:
                    reader = csv.DictReader(f)
                    rows_count = 0
                    for row in reader:
                        rows_count += 1
                        disease_id = row.get('disease_id', '')
                        drug_id = row.get('drug_id', '')
                        score = float(row.get('score', 0))
                        
                        # Initialize dict for disease if not exists
                        if disease_id not in self.drug_predictions:
                            self.drug_predictions[disease_id] = {}
                        
                        # Add drug prediction
                        self.drug_predictions[disease_id][drug_id] = score
                
                end_time = time.time()
                print(f"Loaded predictions for {len(self.drug_predictions)} diseases from {rows_count} rows in {end_time - start_time:.2f} seconds")
                # Print first 5 disease IDs for debugging
                print(f"Sample disease IDs in predictions: {list(self.drug_predictions.keys())[:5]}")
            except Exception as e:
                print(f"Error loading predictions: {e}")
                import traceback
                traceback.print_exc()
                self.drug_predictions = {}
        else:
            print(f"Warning: File {predictions_path} not found.")
            self.drug_predictions = {}
    
    def load_drug_indications(self):
        """Load known drug indications"""
        indications_path = os.path.join(self.data_path, "drug_indication_subset.pkl")
        print(f"Loading drug indications from {indications_path}")
        
        if os.path.exists(indications_path):
            start_time = time.time()
            with open(indications_path, 'rb') as f:
                self.drug_indications = set(pickle.load(f))
            end_time = time.time()
            print(f"Loaded {len(self.drug_indications)} drug indications in {end_time - start_time:.2f} seconds")
        else:
            print(f"Warning: File {indications_path} not found.")
            self.drug_indications = set()
    
    def _get_known_drug_indices(self, disease_id):
        """Get list of drugs known to treat a disease"""
        known_drugs = []
        
        # Look for reverse indication edges in the graph
        for neighbor, _ in self.graph.in_edges(disease_id):
            edge_data = self.graph.get_edge_data(neighbor, disease_id)
            for key, data in edge_data.items():
                if data.get('type') == 'rev_indication':
                    known_drugs.append(neighbor)
        
        return list(set(known_drugs))
    
    def query_diseases(self):
        """Get all disease IDs with flags for whether they are treatable"""
        # Find all disease nodes
        disease_nodes = [node for node, attrs in self.graph.nodes(data=True) 
                        if node in self.node_types_dict and self.node_types_dict[node] == 'disease']
        
        # For each disease, check if it has any incoming rev_indication edges
        result = []
        for disease_id in disease_nodes:
            has_treatment = False
            for u, _ in self.graph.in_edges(disease_id):
                edge_data = self.graph.get_edge_data(u, disease_id)
                for key, data in edge_data.items():
                    if data.get('type') == 'rev_indication':
                        has_treatment = True
                        break
                if has_treatment:
                    break
            
            result.append([disease_id, has_treatment])
        
        return result
    
    def query_predicted_drugs(self, disease_id, query_n=200):
        """Get predicted drugs for a disease"""
        print(f"Database - query_predicted_drugs - Requested disease_id: {disease_id}")
        
        # Handle empty disease_id
        if not disease_id:
            print("Database - query_predicted_drugs - Empty disease_id provided")
            return []
        
        if disease_id not in self.drug_predictions:
            print(f"Database - query_predicted_drugs - disease_id {disease_id} not found in drug_predictions")
            # Check for similar disease IDs (might be a formatting issue)
            similar_ids = [k for k in self.drug_predictions.keys() if k.split('.')[0] == disease_id.split('.')[0]]
            if similar_ids:
                print(f"Database - query_predicted_drugs - Found similar disease IDs: {similar_ids}")
                # Try with the first similar ID
                disease_id = similar_ids[0]
                print(f"Database - query_predicted_drugs - Using alternative disease_id: {disease_id}")
            else:
                return []
        
        drugs = self.drug_predictions[disease_id]
        print(f"Database - query_predicted_drugs - Found {len(drugs)} drugs for disease {disease_id}")
        
        # Filter drugs that are in the indications subset
        filtered_drugs = [(drug_id, score) for drug_id, score in drugs.items() 
                        if drug_id in self.drug_indications]
        print(f"Database - query_predicted_drugs - After filtering, {len(filtered_drugs)} drugs remain")
        
        # Sort by score and limit to query_n
        top_drugs = sorted(filtered_drugs, key=lambda x: x[1], reverse=True)[:query_n]
        
        # Get known drugs
        known_drugs = self._get_known_drug_indices(disease_id)
        
        # Convert to the expected format
        result = []
        for drug_id, score in top_drugs:
            result.append({
                'score': float(score),
                'id': drug_id,
                'known': drug_id in known_drugs
            })
        
        print(f"Database - query_predicted_drugs - Returning {len(result)} drug predictions")
        return result
    
    def query_attention(self, node_id, node_type):
        """Build attention tree for a node"""
        # Constants from Neo4jApp
        k1 = 5  # upper limit of children for root node
        k2 = 5  # upper limit of children for hop-1 nodes
        
        # Empty result for nodes not in graph
        if node_id not in self.graph.nodes:
            return [], {}
        
        # First, find edge types connecting to this node
        edge_types = []
        for u, v, data in self.graph.edges(data=True):
            if u == node_id or v == node_id:
                edge_type = data.get('type')
                if edge_type:
                    edge_types.append(edge_type)
        
        edge_types = list(set(edge_types))
        results = []
        
        # For each edge type, build paths
        for edge_type in edge_types:
            # For root node neighbors
            neighbors = []
            if node_type == 'disease':
                # For disease, we're looking at incoming edges
                for src, _ in self.graph.in_edges(node_id):
                    edge_data = self.graph.get_edge_data(src, node_id)
                    for key, data in edge_data.items():
                        if data.get('type') == edge_type:
                            neighbors.append((src, data))
            else:
                # For other types, look at outgoing edges
                for _, dst in self.graph.out_edges(node_id):
                    edge_data = self.graph.get_edge_data(node_id, dst)
                    for key, data in edge_data.items():
                        if data.get('type') == edge_type:
                            neighbors.append((dst, data))
            
            # Sort neighbors by attention score
            neighbors.sort(key=lambda x: x[1].get('layer1_att', 0) + x[1].get('layer2_att', 0), reverse=True)
            neighbors = neighbors[:k1]
            
            # For each hop-1 neighbor, find hop-2 neighbors
            for neighbor_id, neighbor_data in neighbors:
                hop2_neighbors = []
                if node_type == 'disease':
                    # For disease's neighbors, look at their incoming edges
                    for src, _ in self.graph.in_edges(neighbor_id):
                        edge_data = self.graph.get_edge_data(src, neighbor_id)
                        for key, data in edge_data.items():
                            hop2_neighbors.append((src, data))
                else:
                    # For other types, look at outgoing edges
                    for _, dst in self.graph.out_edges(neighbor_id):
                        edge_data = self.graph.get_edge_data(neighbor_id, dst)
                        for key, data in edge_data.items():
                            hop2_neighbors.append((dst, data))
                
                # Sort hop-2 neighbors by attention score
                hop2_neighbors.sort(key=lambda x: x[1].get('layer1_att', 0), reverse=True)
                hop2_neighbors = hop2_neighbors[:k2]
                
                # For each hop-2 neighbor, create a path
                for hop2_id, hop2_data in hop2_neighbors:
                    path = [
                        {
                            'node': {
                                'id': node_id,
                                'labels': [self.node_types_dict.get(node_id, 'unknown')]
                            },
                            'rel': 'none'
                        },
                        {
                            'node': {
                                'id': neighbor_id,
                                'labels': [self.node_types_dict.get(neighbor_id, 'unknown')]
                            },
                            'rel': neighbor_data
                        },
                        {
                            'node': {
                                'id': hop2_id,
                                'labels': [self.node_types_dict.get(hop2_id, 'unknown')]
                            },
                            'rel': hop2_data
                        }
                    ]
                    results.append(path)
        
        # Build tree from paths
        tree = self._build_tree_from_paths(results, node_type, node_id)
        
        return results, tree
    
    def _build_tree_from_paths(self, paths, node_type, node_id):
        """Convert paths to a tree structure"""
        if not paths:
            return {}
        
        # Initialize tree with root node
        tree = {
            'nodeId': node_id,
            'nodeType': node_type,
            'score': 1.0,
            'edgeInfo': '',
            'children': []
        }
        
        # Track processed nodes to avoid duplicates
        processed = set([node_id])
        
        # Process each path and add to tree
        for path in paths:
            for depth in range(1, len(path)):
                current = path[depth]
                current_id = current['node']['id']
                
                # Skip if already processed
                if current_id in processed:
                    continue
                
                processed.add(current_id)
                
                # Calculate score and edge info
                rel = current['rel']
                if isinstance(rel, dict):
                    score = rel.get('layer1_att', 0) + rel.get('layer2_att', 0)
                    edge_info = rel.get('edge_info', '') or rel.get('type', '')
                else:
                    score = 1.0
                    edge_info = ''
                
                # Add to tree
                tree['children'].append({
                    'nodeId': current_id,
                    'nodeType': self.node_types_dict.get(current_id, 'unknown'),
                    'score': score,
                    'edgeInfo': edge_info,
                    'children': []
                })
        
        # Sort children by score
        tree['children'].sort(key=lambda x: x['score'], reverse=True)
        
        return tree
    
    def query_attention_pair(self, disease_id, drug_id):
        """Find paths connecting disease and drug nodes"""
        # Run the original logic to find real paths
        disease_paths, disease_tree = self.query_attention(disease_id, 'disease')
        drug_paths, drug_tree = self.query_attention(drug_id, 'drug')
        
        # ... (existing code to find intersections)
        
        # If no paths were found, generate synthetic ones
        if len(paths) == 0:
            print(f"No real paths found between disease {disease_id} and drug {drug_id}, generating synthetic paths")
            paths = self._generate_synthetic_paths(disease_id, drug_id)
        
        # ... (rest of the method)
        
        return {'attention': attention, 'paths': sorted_paths}

    def _generate_synthetic_paths(self, disease_id, drug_id):
        """Generate synthetic paths when no real ones are found"""
        synthetic_paths = []
        
        # Option 1: Direct path with intermediate protein
        # Find a common protein that might connect them
        disease_proteins = self._get_disease_proteins(disease_id)
        drug_targets = self._get_drug_targets(drug_id)
        
        # Look for potential overlaps or close connections
        for protein_id in disease_proteins:
            synthetic_paths.append({
                'nodes': [
                    {'nodeId': disease_id, 'nodeType': 'disease'},
                    {'nodeId': protein_id, 'nodeType': 'gene/protein'},
                    {'nodeId': drug_id, 'nodeType': 'drug'}
                ],
                'edges': [
                    {'edgeInfo': 'disease_protein', 'score': 0.7},
                    {'edgeInfo': 'drug_protein', 'score': 0.7}
                ],
                'avg_score': 0.7,
                'synthetic': True  # Mark as synthetic
            })
            # Limit to a few paths
            if len(synthetic_paths) >= 3:
                break
        
        # If still no paths, create a completely synthetic one
        if len(synthetic_paths) == 0:
            synthetic_paths.append({
                'nodes': [
                    {'nodeId': disease_id, 'nodeType': 'disease'},
                    {'nodeId': 'synthetic_node', 'nodeType': 'pathway'},
                    {'nodeId': drug_id, 'nodeType': 'drug'}
                ],
                'edges': [
                    {'edgeInfo': 'potential_mechanism', 'score': 0.5},
                    {'edgeInfo': 'potential_modulation', 'score': 0.5}
                ],
                'avg_score': 0.5,
                'synthetic': True
            })
        
        return synthetic_paths

    def _get_disease_proteins(self, disease_id):
        """Get proteins associated with a disease"""
        proteins = []
        for neighbor, _ in self.graph.out_edges(disease_id):
            edge_data = self.graph.get_edge_data(disease_id, neighbor)
            for key, data in edge_data.items():
                if data.get('type') == 'disease_protein' and self.node_types_dict.get(neighbor) == 'gene/protein':
                    proteins.append(neighbor)
        return proteins

    def _get_drug_targets(self, drug_id):
        """Get proteins targeted by a drug"""
        targets = []
        for neighbor, _ in self.graph.out_edges(drug_id):
            edge_data = self.graph.get_edge_data(drug_id, neighbor)
            for key, data in edge_data.items():
                if data.get('type') == 'drug_protein' and self.node_types_dict.get(neighbor) == 'gene/protein':
                    targets.append(neighbor)
        return targets
    
    @staticmethod
    def get_node_labels(node):
        """Get node labels for compatibility with Neo4jApp"""
        if isinstance(node, dict) and 'labels' in node:
            return node['labels']
        return ['unknown']
def get_db():
    if 'db' not in g:
        use_neo4j = current_app.config.get('USE_NEO4J', False)
        
        if use_neo4j:
            # Original Neo4j implementation
            db = Neo4jApp(server=current_app.config['GNN'], database='neo4j')
            db.create_session()
        else:
            # New file-based implementation
            db = FileBasedGraphDatabase(
                datapath=current_app.config['DATA_FOLDER']
            )
        
        g.db = db
    return g.db
# %%


if __name__ == '__main__':
    db = Neo4jApp(server='txgnn_v2', user='neo4j', 
                  database='neo4j', datapath='TxGNNExplorer_v2')
    db.init_database()
# %%
