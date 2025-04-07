#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Evaluator class."""

from argparse import Namespace

from dctools.data.dataloader import DatasetLoader
from dctools.data.dataset import DCEmptyDataset
from dctools.metrics.evaluator import Evaluator
from dctools.metrics.metrics import MetricComputer
from dctools.data.transforms import CustomTransforms
from dctools.utilities.init_dask import setup_dask
from dctools.utilities.xarray_utils import DICT_RENAME_CMEMS,\
    LIST_VARS_GLONET


class DC2Evaluation:
    """Class to evaluate models."""

    def __init__(self, arguments: Namespace) -> None:
        """Init class.

        Args:
            aruguments (str): Namespace with config.
        """
        self.args = arguments

    def run_eval(self) -> None:
        """Proceed to evaluation."""
        dask_cluster = setup_dask(self.args)
        data_dir1 = self.args.data_dir1
        data_dir2 = self.args.data_dir2

        transf_fct = CustomTransforms(
            transform_name="rename_subset_vars",
            dict_rename=DICT_RENAME_CMEMS,
            list_vars=LIST_VARS_GLONET,
        )
        self.args.dclogger.info("Creating datasets.")
        dataset_1 = DCEmptyDataset(
            conf_args=self.args,
            root_data_dir= data_dir1,
            list_dates=self.args.list_start_dates,
            transform_fct=transf_fct,
        )

        dataset_2 = DCEmptyDataset(
            conf_args=self.args,
            root_data_dir= data_dir2,
            list_dates=self.args.list_start_dates,
            transform_fct=transf_fct,
        )
        # 1. Chargement des données de référence et des prédictions avec DatasetLoader
        data_loader = DatasetLoader(
            pred_dataset=dataset_1,
            ref_dataset=dataset_2
        )

        # 3. Exécution de l’évaluation sur plusieurs modèles
        evaluator = Evaluator(
            self.args,
            dask_cluster=dask_cluster, metrics=None,
            data_container={'glonet': data_loader},
        )

        metrics = [
            MetricComputer(
                dc_logger=self.args.dclogger,
                exc_handler=self.args.exception_handler,
                metric_name='rmse', plot_result=False,
            ),

            MetricComputer(
                dc_logger=self.args.dclogger,
                exc_handler=self.args.exception_handler,
                metric_name='energy_cascad',
                plot_result=False,
                var="uo", depth=2,
            ),
        ]
        ''' TODO : check error on oceanbench : why depth = 0 ? -> crash
            MetricComputer(
                dc_logger=test_vars.dclogger,
                exc_handler=test_vars.exception_handler,
                metric_name='euclid_dist',
                plot_result=True,
                minimum_latitude=0,
                maximum_latitude=10,
                minimum_longitude=0,
                maximum_longitude=10,
            ),'''

        evaluator.set_metrics(metrics)
        self.args.dclogger.info("Run computation of metrics.")
        results = evaluator.evaluate()

        self.args.dclogger.info(f"Computed metrics : {results}")

