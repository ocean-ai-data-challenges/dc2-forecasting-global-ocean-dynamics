#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Evaluation of a model against a given reference."""

import os
import sys

from dctools.utilities.args_config import load_args_and_config

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
        args = load_args_and_config(config_path)
        if args is None:
            print("Config loading failed.")
            return 1

        vars(args)['data_dir1'] = os.path.join(args.data_directory, 'data1')
        vars(args)['data_dir2'] = os.path.join(args.data_directory, 'data2')
        vars(args)['weights_path'] = os.path.join(args.data_directory, 'weights')

        os.makedirs(args.data_dir1, exist_ok=True)
        os.makedirs(args.data_dir2, exist_ok=True)

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
