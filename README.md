# ML Prep and Train Scripts

[![](https://img.shields.io/badge/license-GPLv3-c00404.svg)](https://github.com/alex-feel/ml-prep-and-train/blob/main/LICENSE)

This repository contains a collection of Python scripts designed to streamline data analysis, preprocessing, and binary classification modeling.

> **Note**
> Please note that the scripts in this repository are a work in progress and may not be perfect. I'm aware of potential improvements, and your insights and contributions are welcome! If you find something that could be enhanced or have ideas for new features, feel free to open an issue or submit a clear and concise Pull Request.

## Scripts Overview

### `analyze_dataset.py`

#### Description

The `analyze_dataset.py` script provides functionality to analyze a given CSV file, offering insights into the structure and statistics of the dataset. It covers essential details such as:

* Total number of columns
* Total number of rows
* Number of duplicate rows
* Information about each column, including data type, unique values, missing values, and more

#### Usage

You can run the script from the command line using the following syntax:

```commandline
python analyze_dataset.py <path_to_csv_file> [OPTIONS]
```

#### Output

The script will generate a text file containing detailed information about the dataset's structure and characteristics. The output file will be named according to the source CSV file, appending `_analysis.txt` to the original name. For example, if the analyzed CSV file is named `data.csv`, the output text file will be named `data_analysis.txt`.

#### Additional Help

For more detailed information on the usage of the script and the available options, you can run the script with the --help flag from the command line:

```commandline
python analyze_dataset.py --help
```

### `preprocess_data.py`

#### Description

The `preprocess_data.py` script provides a comprehensive tool to preprocess data according to a user-defined JSON configuration. It performs various data preprocessing tasks, including but not limited to:

* Dropping unnecessary columns
* Handling missing values
* Encoding categorical features
* Scaling numerical features
* Splitting the data into training and testing sets
* Handling imbalanced classes

The script takes a CSV file as input and produces processed training and testing datasets, allowing seamless integration into a machine learning workflow.

#### Usage

You can run the script from the command line using the following syntax:

```commandline
python preprocess_data.py <path_to_csv_file> <path_to_json_config> [OPTIONS]
```

#### Output

The script will create processed training and testing datasets ready for model training. The output files might include:

* Training dataset
* Testing dataset
* Any additional files or visualizations based on the specific functionalities of the script

#### Additional Help

For more detailed information on the usage of the script and the available options, you can run the script with the --help flag from the command line:

```commandline
python preprocess_data.py --help
```

### `create_models.py`

#### Description

The `create_models.py` script is designed to create (train) models for binary classification tasks according to a user-specified JSON configuration. It supports a wide range of classification algorithms, including but not limited to:

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM
* CatBoost
* Neural Networks (using Keras)

The script leverages the configuration provided in the JSON file to set up the desired models with their training details.

#### Usage

You can run the script from the command line using the following syntax:

```commandline
python create_models.py <path_to_train_file> <path_to_test_file> <path_to_json_config>
```

#### Output

The script will create and save the trained models, possibly along with other information such as performance metrics, visualizations, or additional files based on the specific functionalities of the script.

#### Additional Help

For more detailed information on the usage of the script and the available options, you can run the script with the --help flag from the command line:

```commandline
python create_models.py --help
```

## Example Usage & Documentation

While comprehensive documentation for these scripts has not yet been created and is hoped to be developed in the future, I understand the importance of getting started with ease. To facilitate your work with the scripts and provide a hands-on example, please refer to the `examples` folder.

Inside this folder, you will find an additional README file that guides you through a step-by-step example of creating a model based on the well-known "Titanic Survival Datasets" available from [Kaggle](https://www.kaggle.com/datasets/ashishkumarjayswal/titanic-datasets). This example should help illustrate how to utilize the scripts for your own datasets and projects.

The `examples` folder contains:

* README with instructions
* User configuration
* Example dataset

Feel free to explore and adapt this example to your needs, and don't hesitate to reach out with any questions or suggestions.

## Contributing

If you find areas for improvement or have ideas for enhancements, feel free to open an issue or submit a Pull Request!

## Share Your Support

Like the project? Please [give it a star](https://github.com/alex-feel/ml-prep-and-train) ⭐

You can find more about starring [here](https://docs.github.com/en/get-started/exploring-projects-on-github/saving-repositories-with-stars).

## Contributors

<a href="https://github.com/alex-feel/ml-prep-and-train/graphs/contributors"><img src="https://contrib.rocks/image?repo=alex-feel/ml-prep-and-train" /></a>

<sup>Made with [contrib.rocks](https://contrib.rocks).</sup>

## License

GNU General Public License v3.0. See [LICENSE](https://github.com/alex-feel/ml-prep-and-train/blob/main/LICENSE) for full details.
