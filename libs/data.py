import pickle

def load_catalog(catalog_path):
    with open(catalog_path, "rb") as f:
        return set(pickle.load(f))