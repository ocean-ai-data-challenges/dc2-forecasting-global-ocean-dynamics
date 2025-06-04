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

from dctools.data.datasets.dataset import get_dataset_from_config
from dctools.data.datasets.dataloader import EvaluationDataloader
from dctools.data.datasets.dataset_manager import MultiSourceDatasetManager

from dctools.data.transforms import CustomTransforms
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
        self.keep_vars = {}
        for source in self.args.sources:
            source_name = source['dataset']
            self.keep_vars[source_name] = source['keep_variables']

        #logger.remove()  # Supprime les handlers existants
        #logger.add(sys.stderr, level="INFO")  # N'affiche que INFO et plus grave (masque DEBUG)

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
        dataset_alias: str,
        transform_name: str,
    ):
        """Fixture pour configurer les transformations."""

        global_metadata = dataset_manager.get_catalog(dataset_alias).global_metadata
        coords_rename_dict = global_metadata.get("dimensions_rename_dict")
        logger.debug(f"Coords rename dict: {coords_rename_dict}")
        vars_rename_dict= global_metadata.get("variables_rename_dict")
        logger.debug(f"Vars rename dict: {vars_rename_dict}\n\n")
        # glonet_vars = self.args.sources["glonet"]["keep_variables"]

        # Configurer les transformations
        '''glonet_transform = CustomTransforms(
            transform_name="glorys_to_glonet",
            weights_path=self.args.regridder_weights,
            depth_coord_vals=GLONET_DEPTH_VALS,
            interp_ranges=RANGES_GLONET,
        )'''
        match transform_name:
            case "standardize":
                transform = CustomTransforms(
                    transform_name="standardize_dataset",
                    list_vars=self.keep_vars[dataset_alias],
                    coords_rename_dict=coords_rename_dict,
                    vars_rename_dict=vars_rename_dict,
                )
            case _:
                transform = None
        # Configurer les transformations
        """pred_transform = CustomTransforms(
            transform_name="rename_subset_vars",
            dict_rename={"longitude": "lon", "latitude": "lat"},
            list_vars=["uo", "vo", "zos"],
        )

        ref_transform = CustomTransforms(
            transform_name="interpolate",
            interp_ranges={"lat": np.arange(-10, 10, 0.25), "lon": np.arange(-10, 10, 0.25)},
        )"""
        return {"dataset_alias": transform}


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

        '''glorys_dataset_name = "glorys"
        glonet_dataset_name = "glonet"
        glonet_wasabi_dataset_name = "glonet_wasabi"
        glorys_catalog_path = os.path.join(
            self.args.catalog_dir, glorys_dataset_name + ".json"
        )
        glonet_catalog_path = os.path.join(
            self.args.catalog_dir, glonet_dataset_name + ".json"
        )
        glonet_wasabi_catalog_path = os.path.join(
            self.args.catalog_dir, glonet_wasabi_dataset_name + ".json"
        )'''

        for source in self.args.sources:
            source_name = source['dataset']
            file_pattern = source['file_pattern']
            kwargs = {}
            kwargs["source"] = source
            kwargs["root_data_folder"] = self.args.data_directory
            kwargs["root_catalog_folder"] = self.args.catalog_dir
            kwargs["max_samples"] = self.args.max_samples
            kwargs["file_pattern"] = file_pattern

            logger.debug(f"\n\nSetup dataset {source_name}\n\n")
            match source_name:
                case "SST_fields":
                    sst_dataset = get_dataset_from_config(
                        **kwargs
                    )
                case "SSS_fields":
                    sss_dataset = get_dataset_from_config(
                        **kwargs
                    )
                case "jason1":
                    jason1_dataset = get_dataset_from_config(
                        **kwargs
                    )
                case "jason2":
                    jason2_dataset = get_dataset_from_config(
                        **kwargs
                    )
                case "jason3":
                    jason3_dataset = get_dataset_from_config(
                        **kwargs

                    )
                case "saral":
                    saral_dataset = get_dataset_from_config(
                        **kwargs
                    )
                case "swot":
                    saral_dataset = get_dataset_from_config(
                        **kwargs
                    )
                case "argo_velocities":
                    argo_dataset = get_dataset_from_config(
                        **kwargs
                    )
                #case "argo_profiles":
                #    argo_dataset = get_dataset_from_config(
                #        **kwargs
                #    )

        filter_region = gpd.GeoSeries(geometry.Polygon((
            (self.args.min_lon,self.args.min_lat),
            (self.args.min_lon,self.args.max_lat),
            (self.args.max_lon,self.args.min_lat),
            (self.args.max_lon,self.args.max_lat),
            (self.args.min_lon,self.args.min_lat),
            )), crs="EPSG:4326")

        manager = MultiSourceDatasetManager()

        logger.debug(f"Setup datasets manager")
        # Ajouter les datasets avec des alias
        '''manager.add_dataset("sst", sst_dataset)
        manager.add_dataset("sss", sss_dataset)
        manager.add_dataset("jason1", jason1_dataset)
        manager.add_dataset("jason2", jason2_dataset)'''
        manager.add_dataset("jason3", jason3_dataset)
        '''manager.add_dataset("saral", saral_dataset)
        manager.add_dataset("argo", argo_dataset)'''

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
        dask_cluster = setup_dask(self.args)

        transforms = {}
        dataloaders = {}
        metrics_names = {}
        metrics = {}
        evaluators = {}
        models_results = {}
        for alias in dataset_manager.datasets.keys():
            logger.debug(f"\n   \n\nEvaluate dataset {alias}\n\n\n")
            transforms.update(self.setup_transforms(dataset_manager, alias, 'stadardize'))
            # transform_standardize_jason3 = transforms["standardize_jason3"]
            # Créer un dataloader
            """dataloader = manager.get_dataloader(
                pred_alias="glonet",
                ref_alias="glorys",
                batch_size=8,
                pred_transform=glonet_transform,
                ref_transform=glonet_transform,
            )"""
            dataloaders[alias] = dataset_manager.get_dataloader(
                pred_alias=alias,
                ref_alias=alias,
                batch_size=self.args.batch_size,
                pred_transform=transforms.get(alias),
                ref_transform=transforms.get(alias),
            )

            # Vérifier le dataloader
            self.check_dataloader(dataloaders[alias])



        for alias in dataset_manager.datasets.keys():
            metrics_names[alias] = [
                "rmsd",
            ]
            metrics_kwargs = {}
            metrics_kwargs[alias] = {"add_noise": False,
                "eval_variables": dataloaders["jason3"].eval_variables,
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



