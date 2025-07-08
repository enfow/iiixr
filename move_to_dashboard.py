import argparse
import json
import os
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer_dir", type=str, required=True)
    parser.add_argument("--dashboard_dir", type=str, required=True)
    args = parser.parse_args()

    trainer_dir = args.trainer_dir
    dashboard_dir = args.dashboard_dir

    # read config.json
    with open(os.path.join(trainer_dir, "config.json"), "r") as f:
        config = json.load(f)

    model_name = config["model"]["model"]

    os.makedirs(os.path.join(dashboard_dir, model_name), exist_ok=True)

    # copy config.json to dashboard_dir
    shutil.copy(
        os.path.join(trainer_dir, "config.json"),
        os.path.join(dashboard_dir, model_name, f"config.json"),
    )

    # copy the file which has surfix demo.gif, training.png, loss.png, evaluation.png to dashboard_dir
    for file in os.listdir(trainer_dir):
        if file.endswith("demo.gif"):
            shutil.copy(
                os.path.join(trainer_dir, file),
                os.path.join(dashboard_dir, model_name, f"demo.gif"),
            )
        if file.endswith("training.png"):
            shutil.copy(
                os.path.join(trainer_dir, file),
                os.path.join(dashboard_dir, model_name, f"training.png"),
            )
        if file.endswith("loss.png"):
            shutil.copy(
                os.path.join(trainer_dir, file),
                os.path.join(dashboard_dir, model_name, f"loss.png"),
            )
        if file.endswith("evaluation.png"):
            shutil.copy(
                os.path.join(trainer_dir, file),
                os.path.join(dashboard_dir, model_name, f"evaluation.png"),
            )


if __name__ == "__main__":
    main()
