import json
import os

import optuna
from ab.nn.util.Train import Train
from ab.nn.util.Loader import Loader


def ensure_directory_exists(model_dir):
    """
    Ensures that the directory for the given path exists.
    :param model_dir: Path to the target directory or file.
    :return: Creates the directory if it does not exist.
    """
    directory = os.path.dirname(model_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_results(config_model_name, study, n_epochs):
    """
    Save Optuna study results for a given model in JSON-format.
    :param config_model_name: Config (Task, Dataset, Normalization) and Model name.
    :param study: Optuna study object.
    :param n_epochs: Number of epochs.
    """
    model_dir = f"stat/{config_model_name}/{n_epochs}/"
    ensure_directory_exists(model_dir)

    # Save all trials as trials.json
    trials_df = study.trials_dataframe()
    filtered_trials = trials_df[["value", "params_batch_size", "params_lr", "params_momentum"]]

    filtered_trials = filtered_trials.rename(columns={
        "value": "accuracy",
        "params_batch_size": "batch_size",
        "params_lr": "lr",
        "params_momentum": "momentum"
    })

    filtered_trials = filtered_trials.astype({
        "accuracy": float,
        "batch_size": int,
        "lr": float,
        "momentum": float
    })

    trials_dict = filtered_trials.to_dict(orient='records')
    path = f"{model_dir}/trials.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            previous_trials = json.load(f)
            trials_dict = previous_trials + trials_dict

    trials_dict = sorted(trials_dict, key=lambda x: x['accuracy'], reverse=True)
    # Save trials.json
    with open(path, "w") as f:
        json.dump(trials_dict, f, indent=4)

    # Save best_trial.json
    with open(f"{model_dir}/best_trial.json", "w") as f:
        json.dump(trials_dict[0], f, indent=4)

    print(f"Trials for {config_model_name} saved at {model_dir}")


def count_trials_left(config_model_name, model_name, n_epochs, n_optuna_trials):
    """
    Calculates the remaining Optuna trials based on the completed ones. Checks for a "trials.json" file in the
    specified directory to determine how many trials have been completed, and returns the number of trials left.
    :param config_model_name: Configuration model name.
    :param model_name: Name of the model.
    :param n_epochs: Number of epochs.
    :param n_optuna_trials: Total number of Optuna trials.
    :return: n_trials_left: Remaining trials.
    """
    model_dir = f"stat/{config_model_name}/{n_epochs}/"
    path = f"{model_dir}/trials.json"
    n_passed_trials = 0
    if os.path.exists(path):
        with open(path, "r") as f:
            trials = json.load(f)
            n_passed_trials = len(trials)
    n_trials_left = max(0, n_optuna_trials - n_passed_trials)
    if n_passed_trials > 0:
        print(f"The {model_name} passed {n_passed_trials} trials, {n_trials_left} left.")
    return n_trials_left


def extract_all_configs(config):
    """
    Collect models matching the given configuration prefix
    """
    return [sub_config
            for sub_config in os.listdir("stat")
            if sub_config.startswith(config) and os.path.isdir(os.path.join("stat", sub_config))]


# todo: move to the ab.nn.util.Start and request information from database
def provide_all_configs(config):
    if not isinstance(config, list):
        config = [config]
    all_configs = []
    for c in config:
        all_configs = all_configs + extract_all_configs(c)
    return all_configs


def main(config='', n_epochs=1, n_optuna_trials=100):
    """
    Main function for training models using Optuna optimization.
    :param config: Configuration specifying the models to train. The default value for all configurations.
    :param n_epochs: Number or list of numbers of epochs for training.
    :param n_optuna_trials: Number of Optuna trials.
    """

    # Parameters specific to dataset loading.
    dataset_params = {}

    # Determine configurations based on the provided config
    sub_configs = provide_all_configs(config)

    if not isinstance(n_epochs, list):
        n_epochs = [n_epochs]
    for epoch in n_epochs:
        print(f"Configurations found for training for {epoch} epochs:")
        for idx, sub_config in enumerate(sub_configs, start=1):
            print(f"{idx}. {sub_config}")

        for config_name in sub_configs:
            sub_configs = [config_name]

            for sub_config in sub_configs:
                try:
                    task, dataset_name, metric, transform_name, model_name = sub_config.split('-')
                except (ValueError, IndexError) as e:
                    print(f"Skipping config '{sub_config}': failed to parse. Error: {e}")
                    continue

                n_optuna_trials_left = count_trials_left(config_name, model_name, epoch, n_optuna_trials)
                if n_optuna_trials_left == 0:
                    print(f"The model {model_name} has already passed all trials for task: {task}, dataset: {dataset_name},"
                          f" metric: {metric}, transform: {transform_name}, epochs: {epoch}")
                else:
                    print(f"\nStarting training for the model: {model_name}, task: {task}, dataset: {dataset_name},"
                          f" metric: {metric}, transform: {transform_name}, epochs: {epoch}")
                    if task == "img_segmentation" or task == "obj_detection":
                        import ab.nn.loader.coco as coco
                        dataset_params['class_list'] = coco.class_list()
                        dataset_params['path'] = "./data/coco"


                        
                    # Paths for loader and transform
                    loader_path = f"loader.{dataset_name}.loader"
                    transform_path = f"transform.{transform_name}.transform" if transform_name else None

                    # Load dataset
                    try:
                        output_dimension, train_set, test_set = Loader.load_dataset(loader_path, transform_path,
                                                                                    **dataset_params)
                    except Exception as e:
                        print(f"Skipping model '{model_name}': failed to load dataset. Error: {e}")
                        continue

                    # Configure Optuna for the current model
                    def objective(trial):
                        if task == 'img_segmentation':
                            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=False)
                            momentum = trial.suggest_float('momentum', 0.8, 0.99, log=True)
                        else:
                            lr = trial.suggest_float('lr', 1e-4, 1, log=False)
                            momentum = trial.suggest_float('momentum', 0.01, 0.99, log=True)
                        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64])
                        print(f"Initialize training with lr = {lr}, momentum = {momentum}, batch_size = {batch_size}")

                        if task == 'txt_generation':
                            # Dynamically import RNN or LSTM model
                            if model_name.lower() == 'rnn':
                                from dataset.RNN import Net as RNNNet
                                model = RNNNet(1, 256, len(train_set.chars), batch_size)
                            elif model_name.lower() == 'lstm':
                                from ab.nn.dataset.LSTM import Net as LSTMNet
                                model = LSTMNet(1, 256, len(train_set.chars), batch_size, num_layers=2)
                            else:
                                raise ValueError(f"Unsupported text generation model: {model_name}")

                        trainer = Train(
                            model_source_package=f"dataset.{model_name}",
                            task_type=task,
                            train_dataset=train_set,
                            test_dataset=test_set,
                            metric=metric,
                            output_dimension=output_dimension,
                            lr=lr,
                            momentum=momentum,
                            batch_size=batch_size)
                        return trainer.evaluate(epoch)

                    # Launch Optuna for the current model
                    study_name = f"{model_name}_study"
                    study = optuna.create_study(study_name=study_name, direction='maximize')
                    study.optimize(objective, n_trials=n_optuna_trials_left)

                    # Save results
                    save_results(sub_config, study, epoch)


if __name__ == "__main__":
    # Config examples
    conf = '' # For all configurations
    # conf = 'img_classification' # For all image classification configurations
    # conf = 'img_classification-cifar10-acc-cifar10_norm' # For a particular configuration for all models
    # conf = 'img_classification-cifar10-acc-cifar10_norm-GoogLeNet'  # For a particular configuration and model
    # conf = ['img_classification', 'img_segmentation']  # For all image classification and segmentation configurations
    conf = 'obj_detection-coco_detection-map-coco_detection_transform-RetinaNet'
    # Run training with Optuna: detects and saves performance metric values for a varying number of epochs
    main(conf, [1, 2, 5], 10)
