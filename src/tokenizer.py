import json

import hydra
from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from transformers import BertTokenizerFast


def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.rstrip("\n|\r")))
    print("Loaded {} records from {}".format(len(data), input_path))

    data_txt = []
    for sample in data:
        data_txt.append(sample["content"].strip())

    print(data_txt[:3])

    return data_txt


@hydra.main(config_name="config.yml")
def main(cfg):
    # load json
    data_txt = load_jsonl(cfg.PATH.data_dir)

    # bert tokenizer save
    bert_tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    bert_tokenizer.normalizer = normalizers.Sequence(
        [NFD(), Lowercase(), StripAccents()]
    )
    bert_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    bert_tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )

    trainer = trainers.WordPieceTrainer(
        vocab_size=cfg.TOKENIZER.vocab_size,
        show_progress=True,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    )

    bert_tokenizer.train_from_iterator(data_txt, trainer=trainer)
    bert_tokenizer.model.save(cfg.PATH.tokenizer_dir)

    tokenizer_for_load = BertTokenizerFast.from_pretrained(cfg.PATH.tokenizer_dir)  # 로드
    tokenizer_for_load.save_pretrained(cfg.PATH.tokenizer_dir, legacy_format=False)


if __name__ == "__main__":
    main()
