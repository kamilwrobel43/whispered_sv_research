import json
from pathlib import Path

from dataset import ChainsDatasetSV
from utils import split_speakers, generate_embeddings, evaluate_modes
from model import SVModel
import torch
from torch.utils.data import DataLoader
import hydra
from hydra.core.config_store import ConfigStore
from configs.config_classes import Config

cs = ConfigStore.instance()
cs.store(name="base_cfg", node=Config)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: Config):

    solo_path = cfg.data.solo_path
    whsp_path = cfg.data.whsp_path
    split_ratio = cfg.data.split_ratio
    batch_size = cfg.training.batch_size

    results_filename = cfg.baseline.results_filename
    model_names = cfg.baseline.model_names
    results_by_model = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in model_names:
        model = SVModel(model_name=model_name).to(device)
        model.eval()

        per_model_results = {
            "model_name": model_name,
            "seeds": [],
            "num_test_speakers": [],
            "test_speakers": [],
            "results": [],
        }

        for seed in [43]:
            _, test_speakers = split_speakers(root_dir=solo_path, train_ratio=split_ratio, seed=seed)
            test_dataset = ChainsDatasetSV(solo_path, whsp_path, test_speakers)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

            embeddings, speaker_labels, style_labels = generate_embeddings(
                test_loader=test_loader,
                model=model,
                device=device,
            )
            mode_results = evaluate_modes(
                embeddings=embeddings,
                speaker_labels=speaker_labels,
                style_labels=style_labels,
                modes=["neutral-whisper", "neutral-neutral", "whisper-whisper", "all"],
                seed=seed,
            )

            per_model_results["seeds"].append(seed)
            per_model_results["num_test_speakers"].append(len(test_speakers))
            per_model_results["test_speakers"].append(test_speakers)
            per_model_results["results"].append(
                {
                    mode: result["eer"]
                    for mode, result in mode_results.items()
                }
            )

        results_by_model[model_name] = per_model_results

    output_path = Path("results") / results_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results_by_model, f, indent=2)

    print(f"Saved JSON results to {output_path}")


if __name__ == "__main__":
    main()