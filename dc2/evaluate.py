#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Evaluation of a model against a given reference."""

import os
import sys
import yaml
from argparse import Namespace
from pathlib import Path

from dc2.evaluation.evaluation import DC2Evaluation


def main() -> int:
    """Main function.

    Args:
        args (Namespace, optional): Namespace of parsed arguments.

    Returns:
        int: return code.
    """
    try:
        config_name = "dc2"
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'config',
            f"{config_name}.yaml",
        )
        
        # Load configuration directly from YAML file
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Create args namespace
        args = Namespace()
        
        # Apply configuration values
        for key, value in config_data.items():
            setattr(args, key, value)
        
        # Setup data directory at the same level as the main directory
        project_root = Path(__file__).parent.parent  # Go up from dc2/ to project root
        # parent_dir = project_root.parent  # Go up one more level
        data_directory = os.path.join(project_root, "dc2_output")
        
        # Setup paths
        args.data_directory = str(data_directory)
        args.logfile = os.path.join(data_directory, "logs", "dc2.log")

        # Create data directory and subdirectories
        data_directory = Path(args.data_directory)
        logs_dir = data_directory / "logs"
        catalogs_dir = data_directory / "catalogs"
        results_dir = data_directory / "results"
        
        # Create directories
        data_directory.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        catalogs_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Update args with additional paths
        args.regridder_weights = str(data_directory / 'weights')
        args.catalog_dir = str(catalogs_dir)
        args.result_dir = str(results_dir)

        # Clean up existing weights file if it exists
        if os.path.exists(args.regridder_weights):
            os.remove(args.regridder_weights)

        print(f"📁 Output directory: {args.data_directory}")
        print(f"📁 Log file: {args.logfile}")
        print(f"📁 Catalog directory: {args.catalog_dir}")
        print(f"📁 Results directory: {args.result_dir}")

        evaluator_instance = DC2Evaluation(args)
        evaluator_instance.run_eval()
        print("Evaluation has finished successfully.")
        return 0

    except KeyboardInterrupt:
        # raise Exception("Manual abort.")
        print("Manual abort.")
        # Error = non-zero return code
        return 1
    except SystemExit:
        # SystemExit is raised when the user calls sys.exit()
        # or when an error occurs in the argument parsing
        # (e.g. --help)
        # raise Exception("SystemExit.")
        print("SystemExit.")
        # Error = non-zero return code
        return 1

if __name__ == "__main__":
    sys.exit(main())