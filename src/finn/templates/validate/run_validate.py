import os
import sys


def run_validate(dataset, cls_inst, *args, **kwargs):
    dp = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    sys.path.insert(0, dp)
    try:
        import validate.cifar.validate as cifar_validate
        import validate.mnist.validate as mnist_validate
        import validate.radioml.validate as radioml_validate
    finally:
        # Remove the added path to avoid side effects
        if dp in sys.path:
            sys.path.remove(dp)

    print(f"Running validation for dataset: {dataset}")
    print(f"Report directory: {kwargs.get('report_dir')}")

    if dataset == "mnist":
        mnist_validate.validate(cls_inst, *args, **kwargs)
    elif dataset == "cifar":
        cifar_validate.validate(cls_inst, *args, **kwargs)
    elif dataset == "radioml":
        radioml_validate.validate(cls_inst, *args, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
