"""
Utilities for loading configurations, instantiating Python objects, and
running operations in _Selene_.

"""
import os
import importlib
import sys
from time import strftime
import types

import torch
from torch.utils.tensorboard import SummaryWriter

from . import _is_lua_trained_model
from . import instantiate


def class_instantiate(classobj):
    """Not used currently, but might be useful later for recursive
    class instantiation
    """
    for attr, obj in classobj.__dict__.items():
        is_module = getattr(obj, "__module__", None)
        if is_module and "selene_sdk" in is_module and attr is not "model":
            class_instantiate(obj)
    classobj.__init__(**classobj.__dict__)


def module_from_file(path):
    """
    Load a module created based on a Python file path.

    Parameters
    ----------
    path : str
        Path to the model architecture file.

    Returns
    -------
    The loaded module

    """
    parent_path, module_file = os.path.split(path)
    loader = importlib.machinery.SourceFileLoader(module_file[:-3], path)
    module = types.ModuleType(loader.name)
    loader.exec_module(module)
    return module


def module_from_dir(path):
    """
    This method expects that you pass in the path to a valid Python module,
    where the `__init__.py` file already imports the model class,
    `criterion`, and `get_optimizer` methods from the appropriate file
    (e.g. `__init__.py` contains the line `from <model_class_file> import
    <ModelClass>`).

    Parameters
    ----------
    path : str
        Path to the Python module containing the model class.

    Returns
    -------
    The loaded module
    """
    parent_path, module_dir = os.path.split(path)
    sys.path.insert(0, parent_path)
    return importlib.import_module(module_dir)


def initialize_model(model_configs, loss_configs=None, train=True, lr=None):
    """
    Initialize model (and associated criterion, optimizer)

    Parameters
    ----------
    model_configs : dict
        Model-specific configuration
    loss_configs : dict
        Criterion-specific configuration
    train : bool, optional
        Default is True. If `train`, returns the user-specified optimizer
        and optimizer class that can be found within the input model file.
    lr : float or None, optional
        If `train`, a learning rate must be specified. Otherwise, None.

    Returns
    -------
    model, criterion : tuple(torch.nn.Module, torch.nn._Loss) or \
            model, criterion, optim_class, optim_kwargs : \
                tuple(torch.nn.Module, torch.nn._Loss, torch.optim, dict)

        * `torch.nn.Module` - the model architecture
        * `torch.nn._Loss` - the loss function associated with the model
        * `torch.optim` - the optimizer associated with the model
        * `dict` - the optimizer arguments

        The optimizer and its arguments are only returned if `train` is
        True.

    Raises
    ------
    ValueError
        If `train` but the `lr` specified is not a float.

    """
    import_model_from = model_configs["path"]
    model_class_name = model_configs["class"]

    module = None
    if os.path.isdir(import_model_from):
        module = module_from_dir(import_model_from)
    else:
        module = module_from_file(import_model_from)
    model_class = getattr(module, model_class_name)

    model = model_class(**model_configs["class_args"])
    if "non_strand_specific" in model_configs:
        from selene_sdk.utils import NonStrandSpecific

        model = NonStrandSpecific(model, mode=model_configs["non_strand_specific"])

    _is_lua_trained_model(model)
    if loss_configs is not None:
        criterion = instantiate(loss_configs)
        #criterion = module.criterion(**loss_configs)
    else:
        criterion = module.criterion()
    if train and isinstance(lr, float):
        optim_class, optim_kwargs = module.get_optimizer(lr)
        return model, criterion, optim_class, optim_kwargs
    elif train:
        raise ValueError(
            "Learning rate must be specified as a float " "but was {0}".format(lr)
        )
    return model, criterion


def create_data_source(configs, output_dir=None, load_train_val=True, load_test=True):
    """
    Construct data source(s) specified in `configs` (either a data sampler
    or data loader(s)) used in training/evaluation.

    Parameters
    ----------
    configs : dict or object
        The loaded configurations from a YAML file.
    output_dir : str or None
        The path to the directory where all outputs will be saved.
        If None, this means that an `output_dir` was not specified
        in the top-level configuration keys. `output_dir` must be
        specified in each class's individual configuration wherever
        it is required.
    load_train_val: bool
        Return training and validation data loaders.
        Only works when `"dataset" in configs`.
    load_test: bool
        Return test data loader. Only works when `"dataset" in configs`.

    Returns
    -------
    model : Sampler or \
        dataloaders : tuple(torch.utils.data.DataLoader)
        Returns either a single data sampler specified in configs or
        a tuple of data loaders according `load_train_val` and `load_test`,
        which are not mutually exclusive.
    """
    if "sampler" in configs:
        sampler_info = configs["sampler"]
        if output_dir is not None:
            sampler_info.bind(output_dir=output_dir)
        sampler = instantiate(sampler_info)
        return sampler
    if "dataset" in configs:
        dataset_info = configs["dataset"]
        intervals = {
                     "train":[],
                     "validation":[],
                     "test":[]
                     }
        for prefix in intervals.keys():
            if prefix+"_intervals_path" in dataset_info:
                with open(dataset_info[prefix+"_intervals_path"]) as f:
                    for line in f:
                        chrom, start, end = line.rstrip().split("\t")[:3]
                        start = int(start)
                        end = int(end)
                        intervals[prefix].append((chrom, start, end))

        if "sampling_intervals_path" in dataset_info.keys():
            with open(dataset_info["sampling_intervals_path"]) as f:
                for line in f:
                    chrom, start, end = line.rstrip().split("\t")[:3]
                    start = int(start)
                    end = int(end)
                    if load_train_val and chrom in dataset_info["validation_holdout"]:
                        intervals["validation"].append((chrom, start, end))
                    elif load_test and chrom in dataset_info["test_holdout"]:
                        intervals["test"].append((chrom, start, end))
                    elif load_train_val:
                        intervals["train"].append((chrom, start, end))

        with open(dataset_info["distinct_features_path"]) as f:
            distinct_features = list(map(lambda x: x.rstrip(), f.readlines()))

        with open(dataset_info["target_features_path"]) as f:
            target_features = list(map(lambda x: x.rstrip(), f.readlines()))

        module = None
        if os.path.isdir(dataset_info["path"]):
            module = module_from_dir(dataset_info["path"])
        else:
            module = module_from_file(dataset_info["path"])
        dataset_class = getattr(module, dataset_info["class"])
        dataset_info["dataset_args"]["target_features"] = target_features
        dataset_info["dataset_args"]["distinct_features"] = distinct_features

        # create datasets, samplers, and loaders
        tasks = []
        if load_train_val:
            tasks.extend(["train", "validation"])
        if load_test:
            tasks.append("test")

        loaders = []
        for task in tasks:
            # create dataset
            task_config = dataset_info["dataset_args"].copy()
            task_config["intervals"] = intervals[task]
            if task+"_transform" in dataset_info:
                # load transforms
                transform = instantiate(dataset_info[task+"_transform"])
                task_config["transform"] = transform
            task_dataset = dataset_class(**task_config)

            # create sampler
            sampler = None
            if task+"_sampler_class" in dataset_info:
                sampler_class = getattr(module, dataset_info[task+"_sampler_class"])
                if task+"_sampler_args" not in dataset_info:
                    sampler_args = {}
                else:
                    sampler_args = dataset_info[task+"_sampler_args"]
                if not "generator" in sampler_args:
                    gen = torch.Generator()
                    gen.manual_seed(configs["random_seed"])
                    sampler_args["generator"] = gen
                sampler = sampler_class(task_dataset, **sampler_args)

            task_loader = torch.utils.data.DataLoader(
                    task_dataset,
                    batch_size=dataset_info["loader_args"]["batch_size"],
                    num_workers=dataset_info["loader_args"]["num_workers"],
                    worker_init_fn=module.encode_worker_init_fn,
                    sampler=sampler,
                    drop_last=True
                )
            loaders.append(task_loader)

        return loaders


def execute(operations, configs, output_dir):
    """
    Execute operations in _Selene_.

    Parameters
    ----------
    operations : list(str)
        The list of operations to carry out in _Selene_.
    configs : dict or object
        The loaded configurations from a YAML file.
    output_dir : str or None
        The path to the directory where all outputs will be saved.
        If None, this means that an `output_dir` was not specified
        in the top-level configuration keys. `output_dir` must be
        specified in each class's individual configuration wherever
        it is required.

    Returns
    -------
    None
        Executes the operations listed and outputs any files
        to the dirs specified in each operation's configuration.

    Raises
    ------
    ValueError
        If an expected key in configuration is missing.

    """
    model = None
    train_model = None
    if "dataset" in configs:
        if "train" in operations and "evaluate" in operations:
            train_loader, val_loader, test_loader = create_data_source(configs,
                                                                       output_dir)
        elif "train" in operations:
            train_loader, val_loader = create_data_source(configs, output_dir,
                                                          load_test=False)
        elif "evaluate" in operations:
            test_loader = create_data_source(configs, load_train_val=False,
                                             load_test=True)
    for op in operations:
        if op == "train":
            # make sure we provided the right dimensions in the config
            if (
                "dataset" in configs
                and "n_cell_types" in configs["model"]["class_args"]
            ):
                model_n_cell_types = configs["model"]["class_args"]["n_cell_types"]
                dataset_n_cell_types = train_loader.dataset.n_cell_types
                assert model_n_cell_types == dataset_n_cell_types, f"Expected {dataset_n_cell_types} "\
                    f"cell types based on dataset, got {model_n_cell_types} in config"
                
            if (
                "dataset" in configs
                and "n_genomic_features" in configs["model"]["class_args"]
            ):
                model_n_features = configs["model"]["class_args"]["n_genomic_features"]
                dataset_n_features = train_loader.dataset.n_target_features
                assert model_n_features == dataset_n_features, f"Expected {dataset_n_features} "\
                    f"target features based on dataset, but got {model_n_features} in config"

            # load model, criterion, and optimizer
            if "criterion" in configs:
                loss_configs = configs["criterion"]
            else:
                loss_configs = None
            model, loss, optim, optim_kwargs = initialize_model(
                configs["model"], loss_configs, train=True, lr=configs["lr"]
            )

            # load lr scheduler
            if "lr_scheduler" in configs:
                scheduler_class = configs["lr_scheduler"]["class"]
                scheduler_kwargs = configs["lr_scheduler"]["class_args"]
            else:
                scheduler_class = None
                scheduler_kwargs = None

            # instantiate model training class
            train_model_info = configs["train_model"]
            if output_dir is not None:
                train_model_info.bind(output_dir=output_dir)

            if "sampler" in configs:
                sampler = create_data_source(configs, output_dir)
                train_model_info.bind(
                    model=model,
                    data_sampler=sampler,
                    loss_criterion=loss,
                    optimizer_class=optim,
                    optimizer_kwargs=optim_kwargs,
                )
            if "dataset" in configs:
                train_model_info.bind(
                    model=model,
                    loss_criterion=loss,
                    optimizer_class=optim,
                    optimizer_kwargs=optim_kwargs,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    scheduler_class=scheduler_class,
                    scheduler_kwargs=scheduler_kwargs,
                )
            train_model = instantiate(train_model_info)
            # TODO: will find a better way to handle this in the future
            if (
                "sampler" in configs
                and "load_test_set" in configs
                and configs["load_test_set"]
                and "evaluate" in operations
            ):
                train_model.create_test_set()
            train_model.train_and_validate()

        elif op == "evaluate":
            if train_model is not None:
                hparam_dict = configs["model"]["class_args"].copy()
                
                if "dataset" in configs:
                    average_scores, _ = train_model.evaluate(test_loader)
                    hparam_dict.update(
                        {"lr": configs["lr"], "steps": train_model.n_epochs}
                    )
                else:
                    average_scores, _ = train_model.evaluate()
                    hparam_dict.update(
                        {"lr": configs["lr"], "steps": train_model.max_steps}
                    )
                with SummaryWriter(os.path.join(output_dir)) as w:
                    w.add_hparams(hparam_dict, average_scores)

            if not model:
                model, loss = initialize_model(configs["model"], train=False)
            if "evaluate_model" in configs:
                evaluate_model_info = configs["evaluate_model"]
                if output_dir is not None:
                    evaluate_model_info.bind(output_dir=output_dir)

                if "sampler" in configs:
                    sampler = create_data_source(configs, output_dir)
                    evaluate_model_info.bind(
                        model=model, criterion=loss, data_sampler=sampler
                    )
                if "dataset" in configs:
                    evaluate_model_info.bind(
                        model=model,
                        criterion=loss,
                        data_loader=test_loader,
                    )

                evaluate_model = instantiate(evaluate_model_info)
                evaluate_model.evaluate()

        elif op == "analyze":
            if not model:
                model, _ = initialize_model(configs["model"], train=False)
            analyze_seqs_info = configs["analyze_sequences"]
            analyze_seqs_info.bind(model=model)

            analyze_seqs = instantiate(analyze_seqs_info)
            if "variant_effect_prediction" in configs:
                vareff_info = configs["variant_effect_prediction"]
                if "vcf_files" not in vareff_info:
                    raise ValueError(
                        "variant effect prediction requires "
                        "as input a list of 1 or more *.vcf "
                        "files ('vcf_files')."
                    )
                for filepath in vareff_info.pop("vcf_files"):
                    analyze_seqs.variant_effect_prediction(filepath, **vareff_info)
            if "in_silico_mutagenesis" in configs:
                ism_info = configs["in_silico_mutagenesis"]
                if "sequence" in ism_info:
                    analyze_seqs.in_silico_mutagenesis(**ism_info)
                elif "input_path" in ism_info:
                    analyze_seqs.in_silico_mutagenesis_from_file(**ism_info)
                elif "fa_files" in ism_info:
                    for filepath in ism_info.pop("fa_files"):
                        analyze_seqs.in_silico_mutagenesis_from_file(
                            filepath, **ism_info
                        )
                else:
                    raise ValueError(
                        "in silico mutagenesis requires as input "
                        "the path to the FASTA file "
                        "('input_path') or a sequence "
                        "('input_sequence') or a list of "
                        "FASTA files ('fa_files'), but found "
                        "neither."
                    )
            if "prediction" in configs:
                predict_info = configs["prediction"]
                analyze_seqs.get_predictions(**predict_info)


def parse_configs_and_run(configs, configs_path, create_subdirectory=True, lr=None):
    """
    Method to parse the configuration YAML file and run each operation
    specified.

    Parameters
    ----------
    configs : dict
        The dictionary of nested configuration parameters. Will look
        for the following top-level parameters:

            * `ops`: A list of 1 or more of the values \
            {"train", "evaluate", "analyze"}. The operations specified\
            determine what objects and information we expect to parse\
            in order to run these operations. This is required.
            * `output_dir`: Output directory to use for all the operations.\
            If no `output_dir` is specified, assumes that all constructors\
            that will be initialized (which have their own configurations\
            in `configs`) have their own `output_dir` specified.\
            Optional.
            * `random_seed`: A random seed set for `torch` and `torch.cuda`\
            for reproducibility. Optional.
            * `lr`: The learning rate, if one of the operations in the list is\
            "train".
            * `load_test_set`: If `ops: [train, evaluate]`, you may set\
               this parameter to True if you would like to load the test\
               set into memory ahead of time--and therefore save the test\
               data to a .bed file at the start of training. This is only\
               useful if you have a machine that can support a large increase\
               (on the order of GBs) in memory usage and if you want to\
               create a test dataset early-on because you do not know if your\
               model will finish training and evaluation within the allotted\
               time that your job is run.

    create_subdirectory : bool, optional
        Default is True. If `create_subdirectory`, will create a directory
        within `output_dir` with the name formatted as "%Y-%m-%d-%H-%M-%S",
        the date/time this method was run.
    lr : float or None, optional
        Default is None. If "lr" (learning rate) is already specified as a
        top-level key in `configs`, there is no need to set `lr` to a value
        unless you want to override the value in `configs`. Otherwise,
        set `lr` to the desired learning rate if "train" is one of the
        operations to be executed.

    Returns
    -------
    None
        Executes the operations listed and outputs any files
        to the dirs specified in each operation's configuration.

    """
    operations = configs["ops"]

    if "train" in operations and "lr" not in configs and lr and lr != "None":
        configs["lr"] = float(lr)
    elif "train" in operations and "lr" in configs and lr and lr != "None":
        print(
            "Warning: learning rate specified in both the "
            "configuration dict and this method's `lr` parameter. "
            "Using the `lr` value input to `parse_configs_and_run` "
            "({0}, not {1}).".format(lr, configs["lr"])
        )

    current_run_output_dir = None
    if "output_dir" not in configs and (
        "train" in operations or ("evaluate" in operations and "sampler" in configs) 
    ):
        print(
            "No top-level output directory specified. All constructors "
            "to be initialized (e.g. Sampler, TrainModel) that require "
            "this parameter must have it specified in their individual "
            "parameter configuration."
        )
    elif "output_dir" in configs:
        current_run_output_dir = configs["output_dir"]
        os.makedirs(current_run_output_dir, exist_ok=True)
        if "create_subdirectory" in configs:
            create_subdirectory = configs["create_subdirectory"]
        if create_subdirectory:
            current_run_output_dir = os.path.join(
                current_run_output_dir, strftime("%Y-%m-%d-%H-%M-%S")
            )
            os.makedirs(current_run_output_dir)
        print("Outputs and logs saved to {0}".format(current_run_output_dir))

    if "random_seed" in configs:
        seed = configs["random_seed"]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        print(
            "Warning: no random seed specified in config file. "
            "Using a random seed ensures results are reproducible."
        )

    if "train" in operations:
        writer = SummaryWriter(os.path.join(current_run_output_dir))
        with open(configs_path, "r") as config_file:
            # Add <pre> to persist spaces
            config_content = "<pre>" + config_file.read() + "</pre>"
            writer.add_text("config", config_content)

        with open(configs["model"]["path"], "r") as model_file:
            # Add <pre> to persist spaces
            model_file_content = "<pre>" + model_file.read() + "</pre>"
            writer.add_text("model", model_file_content)
        writer.close()

    execute(operations, configs, current_run_output_dir)
