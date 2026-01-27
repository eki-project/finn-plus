import os
import sys


def run_validate(validation_dataset, cls_inst, *args, **kwargs):
    dp = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    sys.path.insert(0, dp)
    try:
        import validate.cifar.validate as cifar_validate
        import validate.imagenet.validate as imagenet_validate
        import validate.mnist.validate as mnist_validate
        import validate.radioml.validate as radioml_validate
        import validate.unswnb15.validate as unswnb15_validate
    finally:
        # Remove the added path to avoid side effects
        if dp in sys.path:
            sys.path.remove(dp)

    print(f"Running validation with Dataset: {validation_dataset}")
    print(f"Report directory: {kwargs.get('report_dir')}")

    if validation_dataset == "mnist":
        mnist_validate.validate(cls_inst, *args, **kwargs)
    elif validation_dataset == "cifar":
        cifar_validate.validate(cls_inst, *args, **kwargs)
    elif validation_dataset == "radioml":
        radioml_validate.validate(cls_inst, *args, **kwargs)
    elif validation_dataset == "imagenet":
        imagenet_validate.validate(cls_inst, *args, **kwargs)
    elif validation_dataset == "unswnb15":
        unswnb15_validate.validate(cls_inst, *args, **kwargs)
    else:
        raise ValueError(f"Unknown or no validation Dataset: {validation_dataset}")
