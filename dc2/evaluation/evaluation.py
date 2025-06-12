#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Evaluator class using Glorys forecasts as reference."""

from argparse import Namespace
import os
import sys
from typing import Optional

import geopandas as gpd
from loguru import logger
import pandas as pd
from shapely import geometry
from torchvision import transforms

from dctools.data.datasets.dataset import get_dataset_from_config
from dctools.data.datasets.dataloader import EvaluationDataloader
from dctools.data.datasets.dataset_manager import MultiSourceDatasetManager

from dctools.metrics.evaluator import Evaluator
from dctools.metrics.metrics import MetricComputer
from dctools.utilities.init_dask import setup_dask
from dctools.data.coordinates import (
    RANGES_GLONET,
    GLONET_DEPTH_VALS,
)


class DC2Evaluation:
    """Class to evaluate models on Glorys forecasts."""

    def __init__(self, arguments: Namespace) -> None:
        """Init class.

        Args:
            aruguments (str): Namespace with config.
        """
        self.args = arguments

    def filter_data(
        self, manager: MultiSourceDatasetManager,
        filter_region: gpd.GeoSeries,
    ):
        # Appliquer les filtres temporels
        manager.filter_all_by_date(
            start=pd.to_datetime(self.args.start_times[0]),
            end=pd.to_datetime(self.args.end_times[0]),
        )
        # Appliquer les filtres spatiaux
        manager.filter_all_by_region(
            region=filter_region
        )
        # Appliquer les filtres sur les variables
        #manager.filter_all_by_variable(variables=self.args.target_vars)
        return manager

    def setup_transforms(
        self,
        dataset_manager: MultiSourceDatasetManager,
        aliases: list[str],
    ):
        """Fixture pour configurer les transformations."""
        transforms_dict = {}
        if "jason3" in aliases:
            logger.warning("Jason3 dataset is not available, skipping its transform setup.")
            transforms_dict["jason3"] = dataset_manager.get_transform(
                "standardize",
                dataset_alias="jason3",
            )
        if "glonet" in aliases:
            transforms_dict["glonet"] = dataset_manager.get_transform(
                "standardize",
                dataset_alias="glonet",
            )
            '''glonet_transform_1 = dataset_manager.get_transform(
                "standardize",
                dataset_alias="glonet",
            )
            glonet_transform_2 = dataset_manager.get_transform(
                "glorys_to_glonet",
                dataset_alias="glonet",
                regridder_weights=self.args.regridder_weights,
            )
            transforms_dict["glonet"] = transforms.Compose([
                glonet_transform_1,
                glonet_transform_2,
            ])'''

        return transforms_dict


    def check_dataloader(
        self,
        dataloader: EvaluationDataloader,
    ):
        for batch in dataloader:
            logger.debug(f"Batch: {batch}")
            # Vérifier que le batch contient les clés attendues
            assert "pred_data" in batch[0]
            assert "ref_data" in batch[0]
            # Vérifier que les données sont de type str (paths)
            assert isinstance(batch[0]["pred_data"], str)
            if batch[0]["ref_data"]:
                assert isinstance(batch[0]["ref_data"], str)

    def setup_dataset_manager(self) -> None:

        manager = MultiSourceDatasetManager()
        datasets = {}
        for source in self.args.sources:
            source_name = source['dataset']
            if source_name != "glonet" and source_name != "jason3":
                logger.warning(f"Dataset {source_name} is not supported yet, skipping.")
                continue
            kwargs = {}
            kwargs["source"] = source
            kwargs["root_data_folder"] = self.args.data_directory
            kwargs["root_catalog_folder"] = self.args.catalog_dir
            kwargs["max_samples"] = self.args.max_samples

            logger.debug(f"\n\nSetup dataset {source_name}\n\n")
            datasets[source_name] = get_dataset_from_config(
                **kwargs
            )
            # Ajouter les datasets avec des alias
            manager.add_dataset(source_name, datasets[source_name])

        filter_region = gpd.GeoSeries(geometry.Polygon((
            (self.args.min_lon,self.args.min_lat),
            (self.args.min_lon,self.args.max_lat),
            (self.args.max_lon,self.args.min_lat),
            (self.args.max_lon,self.args.max_lat),
            (self.args.min_lon,self.args.min_lat),
            )), crs="EPSG:4326")

        # Construire le catalogue
        logger.debug(f"Build catalog")
        manager.build_catalogs()
        manager.all_to_json(output_dir=self.args.catalog_dir)

        # Appliquer les filtres temporels
        manager = self.filter_data(manager, filter_region)
        return manager


    def run_eval(self) -> None:
        """Proceed to evaluation."""
        dataset_manager = self.setup_dataset_manager()
        aliases = dataset_manager.datasets.keys()
        dask_cluster = setup_dask(self.args)

        dataloaders = {}
        metrics_names = {}
        metrics = {}
        evaluators = {}
        models_results = {}
        transforms_dict = self.setup_transforms(dataset_manager, aliases)

        for alias in dataset_manager.datasets.keys():
            logger.debug(f"\n\n\nGet dataloader for {alias}")
            logger.debug(f"Transform: {transforms_dict.get(alias)}\n\n\n")
            pred_transform = transforms_dict.get(alias)
            if alias != 'glonet':
                ref_transform = transforms_dict.get(alias)
                ref_alias=alias
            else:
                ref_transform = None
                ref_alias=None
            dataloaders[alias] = dataset_manager.get_dataloader(
                pred_alias=alias,
                ref_alias=ref_alias,
                batch_size=self.args.batch_size,
                pred_transform=pred_transform,
                ref_transform=ref_transform,
            )

            # Vérifier le dataloader
            self.check_dataloader(dataloaders[alias])

        for alias in dataset_manager.datasets.keys():
            metrics_names[alias] = [
                "rmsd",
            ]
            metrics_kwargs = {}
            metrics_kwargs[alias] = {"add_noise": False,
                "eval_variables": dataloaders[alias].eval_variables,
            }
            metrics[alias] = [
                MetricComputer(metric_name=metric, **metrics_kwargs[alias])
                for metric in metrics_names[alias]
            ]

            evaluators[alias] = Evaluator(
                dask_cluster=dask_cluster,
                metrics=metrics[alias],
                dataloader=dataloaders[alias],
                json_path=os.path.join(self.args.catalog_dir, f"test_results_{alias}.json"),
            )

            models_results[alias] = evaluators[alias].evaluate()


        # Vérifier que chaque résultat contient les champs attendus, afficher
        for dataset_alias, results in models_results.items():
            # Vérifier que les résultats existent
            assert len(results) > 0
            logger.info(f"\n\n\nResults for {dataset_alias}:")
            for result in results:
                assert "date" in result
                assert "metric" in result
                assert "result" in result
                logger.info(f"Test Result: {result}")




