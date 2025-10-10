import math

def best_factors(n):
        # Finds a pair (a, b) such that a*b = n and |a - b| is minimized
        for i in range(int(math.sqrt(n)), 0, -1):
            if n % i == 0:
                return (i, n // i)

def compute_tile_bounds(size, splits, index):
    base = size // splits
    rem = size % splits
    start = index * base + min(index, rem)
    end = start + base + (1 if index < rem else 0)
    return start, end

def safe_label(s):
    s = str(s)
    return s.translate(str.maketrans({
        "\\": "\\\\",
        '"': '\\"',
        "<": "\\<",
        ">": "\\>",
        "{": "\\{",
        "}": "\\}",
    }))

class TensorsNeededTracker:
    """ Helper class for tracking the tensors needed for each operation from the execution graph, used within the analysis phase """
    def __init__(self):
        self._data = {}

    def increase_count(self, tensor_id):
        if tensor_id not in self._data:
            self._data[tensor_id] = {"count": 0, "tiles": {}}
        self._data[tensor_id]["count"] += 1

    def decrease_count(self, tensor_id):
        assert tensor_id in self._data, f"Trying to decrease count for unknown tensor_id: {tensor_id}"
        self._data[tensor_id]["count"] -= 1
        if self._data[tensor_id]["count"] <= 0:
            del self._data[tensor_id]

    def add_tile(self, tensor_id, mem_name, tile_key):
        assert tensor_id in self._data, f"Tensor ID '{tensor_id}' must be registered before adding tiles! Check execution graph creation!"
        tiles = self._data[tensor_id]["tiles"]
        if mem_name not in tiles:
            tiles[mem_name] = {}
        mem_tiles = tiles[mem_name]
        mem_tiles[tile_key] = mem_tiles.get(tile_key, 0) + 1

    def remove_tile(self, tensor_id, mem_name, tile_key):
        assert tensor_id in self._data, f"Tensor ID '{tensor_id}' must be registered before removing tiles! Check execution graph creation!"
        tiles = self._data[tensor_id]["tiles"]
        if mem_name in tiles:
            mem_tiles = tiles[mem_name]
            if tile_key in mem_tiles:
                mem_tiles[tile_key] -= 1
                if mem_tiles[tile_key] <= 0:
                    del mem_tiles[tile_key]
            if not mem_tiles:
                del tiles[mem_name]
    
    def has_tile(self, tensor_id, mem_name, tile_key):
        return (tensor_id in self._data and mem_name in self._data[tensor_id]["tiles"] and tile_key in self._data[tensor_id]["tiles"][mem_name] and self._data[tensor_id]["tiles"][mem_name][tile_key] > 0)
    
    def has_tensor(self, tensor_id):
        return tensor_id in self._data

    def is_in_memory(self, tensor_id, mem_name):
        return (tensor_id in self._data and "tiles" in self._data[tensor_id] and mem_name in self._data[tensor_id]["tiles"])

    def get(self):
        assert len(self._data) == len(set(self._data.keys())), "Duplicate keys? That shouldn't be possible!"
        return self._data
