from datasets import load_dataset

def get_dataset(data_args, model_args):
    if data_args.dataset_name is not None:
        if 'c4' not in data_args.dataset_name:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.cache_dir,
                # use_auth_token=True if model_args.use_auth_token else None,
            )
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                    # # use_auth_token=True if model_args.use_auth_token else None,
                )
                raw_datasets["train"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=model_args.cache_dir,
                    # # use_auth_token=True if model_args.use_auth_token else None,
                )

        else:
            raw_datasets = load_dataset(
                'allenai/c4',
                data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
                cache_dir=model_args.cache_dir,
                # # use_auth_token=True if model_args.use_auth_token else None,
            )
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    'allenai/c4',
                    data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                    split=f"validation[:1%]",
                    cache_dir=model_args.cache_dir,
                    # # use_auth_token=True if model_args.use_auth_token else None,
                )
                raw_datasets["train"] = load_dataset(
                    'allenai/c4',
                    data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
                    split=f"train[:1%]",
                    cache_dir=model_args.cache_dir,
                    # # use_auth_token=True if model_args.use_auth_token else None,
                )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            # # use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                # # use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                # # use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )

    return raw_datasets

