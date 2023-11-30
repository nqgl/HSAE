import transformers
from datasets import load_dataset
d = load_dataset("wikitext", "wikitext-103-raw-v1")
d1 = load_dataset("spacerini/gpt2-outputs")
# d2 = load_dataset("monology/pile-uncopyrighted", cache_dir="/media/g/Crucial X6/hf_data")
d3 = load_dataset("roneneldan/TinyStories")
