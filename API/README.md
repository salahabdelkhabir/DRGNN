# DRGNN Web Interface (Fork of Drug_Explorer)

This directory contains the web interface for the DRGNN platform.

## Attribution

This frontend and API server are adapted from:

- **Original Project**: [Drug_Explorer](https://github.com/wangqianwen0418/Drug_Explorer) by Qianwen Wang
- **Original Paper**: "A visual explanation interface for GNN-based drug repurposing"
- **License**: MIT (same as original)

Modifications were made to integrate with the DRGNN model backend and file-based graph database (replacing Neo4j dependency).

## Code Structure

- `drug_server/` - Python Flask backend API
- `src/` - React/TypeScript frontend
- `txgnn_data_v2/` - Pre-computed prediction data and graph structures

## Quick Start

See main [README.md](../README.md) for full setup instructions.
