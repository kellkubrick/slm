import os
from abc import ABC, abstractmethod
from typing import Union, List

import mlflow
import neptune
from dotenv import load_dotenv

from utils.common_functions import convert_lists_and_tuples_to_string


class BaseLogger(ABC):
    """A base experiment logger class."""

    @abstractmethod
    def __init__(self, config):
        """Logs git commit id, dvc hash, environment."""
        pass

    @abstractmethod
    def log_hyperparameters(self, params: dict):
        pass

    @abstractmethod
    def save_metrics(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_plot(self, *args, **kwargs):
        pass

    @abstractmethod
    def stop(self):
        pass


class NeptuneLogger(BaseLogger):
    """A neptune.ai experiment logger class."""

    def __init__(self, config):
        super().__init__(config)
        load_dotenv(config.env_path)

        self.run = neptune.init_run(
            project=config.project,
            api_token=os.environ['NEPTUNE_API_TOKEN'],
            name=config.experiment_name,
            dependencies=config.dependencies_path,
            with_id=config.run_id
        )

    def log_hyperparameters(self, params: dict):
        """Model hyperparameters logging."""
        self.run['hyperparameters'] = convert_lists_and_tuples_to_string(params)

    def save_metrics(self, type_set, metric_name: Union[List[str], str],
                     metric_value: Union[List[float], List[str], float, str], step=None):
        if isinstance(metric_name, List):
            for p_n, p_v in zip(metric_name, metric_value):
                self.run[f"{type_set}/{p_n}"].log(p_v, step=step)
        else:
            self.run[f"{type_set}/{metric_name}"].log(metric_value, step=step)

    def save_plot(self, type_set, plot_name, plt_fig):
        self.run[f"{type_set}/{plot_name}"].append(plt_fig)

    def stop(self):
        self.run.stop()


class MLFlowLogger(BaseLogger):
    def __init__(self, config):
        super().__init__(config)
        if config.tracking_uri:
            mlflow.set_tracking_uri(config.tracking_uri)

        self._init_experiment(config)

    def _init_experiment(self, config):
        """Set up experiment configurations to log."""
        mlflow.set_experiment(config.experiment_name or "default_experiment")

        if config.run_id:
            mlflow.start_run(run_id=config.run_id)
        else:
            mlflow.start_run()

        mlflow.log_artifact(config.dataset_version)
        mlflow.log_artifact(config.dataset_preprocessing)
        mlflow.log_artifact(config.dependencies_path)

    def log_hyperparameters(self, params: dict):
        mlflow.log_params(params)

    def save_metrics(self, type_set, metric_name: Union[List[str], str], metric_value: Union[List[float], float], step):
        if isinstance(metric_name, List):
            for p_n, p_v in zip(metric_name, metric_value):
                mlflow.log_metric(f"{type_set}_{p_n}", p_v, step)
        else:
            mlflow.log_metric(f"{type_set}_{metric_name}", metric_value, step)

    def save_plot(self, type_set, plot_name, plt_fig):
        plot_path = f"temp_plots/{type_set}_{plot_name}.png"
        plt_fig.savefig(plot_path)
        mlflow.log_artifact(plot_path, artifact_path="plots")

    def stop(self):
        mlflow.end_run()