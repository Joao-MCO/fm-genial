import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASES_DIR = os.path.join(BASE_DIR, "databases")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

STATS_DIR = os.path.join(DATABASES_DIR, "stats")
ATTRIBUTES_DIR = os.path.join(DATABASES_DIR, "attributes")