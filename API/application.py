import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'drug_server'))

from drug_server.application import create_app, application as app
