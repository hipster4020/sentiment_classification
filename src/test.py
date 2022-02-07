import hydra
from transformers import AutoTokenizer


@hydra.main(config_name="config.yml")
def main(cfg):
    t = AutoTokenizer.from_pretrained(cfg.PATH.save_dir)
    e = t("본 고안은 이러한 특성을 이용해 사용한다.")
    print(e)
    print(t.decode(e["input_ids"]))


if __name__ == "__main__":
    main()
