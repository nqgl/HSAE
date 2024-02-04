from pathlib import Path
from datasets import load_dataset
from transformer_lens import HookedTransformer
import torch
import einops

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.float16,
    "bfp16": torch.bfloat16,
}
SAVE_DIR = Path.home() / "workspace"
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir()


def get_model(cfg):
    model = (
        HookedTransformer.from_pretrained(cfg.model_name)
        .to(DTYPES[cfg.enc_dtype])
        .to(cfg.device)
    )
    return model


def shuffle_documents(all_tokens):  # assuming the shape[0] is documents
    # print("Shuffled data")
    return all_tokens[torch.randperm(all_tokens.shape[0])]


def load_data(model: HookedTransformer, dataset="NeelNanda/c4-code-tokenized-2b"):
    import os

    reshaped_name = dataset.split("/")[-1] + "_reshaped.pt"
    dataset_reshaped_path = SAVE_DIR / "data" / reshaped_name
    # if dataset exists loading_data_first_time=False
    loading_data_first_time = not dataset_reshaped_path.exists()

    print("first time:", loading_data_first_time)
    if loading_data_first_time:
        data = load_dataset(dataset, split="train", cache_dir=SAVE_DIR / "cache/")
        # data.save_to_disk(os.path.join(SAVE_DIR / "data/", dataset.split("/")[-1]+".hf"))
        if "tokens" not in data.column_names:
            if "text" in data.column_names:
                data.set_format(type="torch", columns=["text"])
                data = data["text"]
                # model.tokenizer.
                all_tokens = model.tokenizer.tokenize(
                    data["text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                )
        else:
            data.set_format(type="torch", columns=["tokens"])
            all_tokens = data["tokens"]
        all_tokens.shape

        all_tokens_reshaped = einops.rearrange(
            all_tokens, "batch (x seq_len) -> (batch x) seq_len", x=8, seq_len=128
        )
        all_tokens_reshaped[:, 0] = model.tokenizer.bos_token_id
        all_tokens_reshaped = all_tokens_reshaped[
            torch.randperm(all_tokens_reshaped.shape[0])
        ]
        print("saving to:", dataset_reshaped_path)
        torch.save(all_tokens_reshaped, dataset_reshaped_path)
        print("saved reshaped data")
    else:
        # data = datasets.load_from_disk("/workspace/data/c4_code_tokenized_2b.hf")
        all_tokens = torch.load(dataset_reshaped_path)
        all_tokens = shuffle_documents(all_tokens)
    return all_tokens
