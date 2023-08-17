import argparse
import os

import numpy as np
import pandas as pd


# Analyze a CSV file
def analyze_csv_file(file_path, unique_values_threshold=10, delimiter=',', output_file_path='dataset_analysis.txt'):
    # Reading CSV file
    data = pd.read_csv(file_path, delimiter=delimiter, encoding='UTF-8')

    # Getting information about columns
    num_columns = len(data.columns)
    # Getting information about rows
    num_rows = len(data)
    # Getting number of duplicate rows
    duplicate_rows = data.duplicated().sum()
    duplicate_rows_percent = duplicate_rows / num_rows * 100

    # Writing column and row information to a text file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(f'Total number of columns: {num_columns}\n')
        f.write(f'Total number of rows: {num_rows}\n')
        f.write(
            f'Number of duplicate rows: {duplicate_rows} ({duplicate_rows_percent:.2f}%)\n\n')

        for column in data.columns:
            column_type = data[column].dtype
            unique_values = data[column].unique()
            num_unique_values = len(unique_values)

            # Checking for missing values
            missing_values = data[column].isnull().sum()
            missing_values_percent = missing_values / num_rows * 100

            # Mode calculation for any type
            mode_value = data[column].mode(
            ).iloc[0] if not data[column].mode().empty else None

            if pd.api.types.is_numeric_dtype(column_type):
                # Checking for zero values
                zero_values = (data[column] == 0).sum()
                zero_values_percent = zero_values / num_rows * 100

                # Checking for negative values
                negative_values = (data[column] < 0).sum()
                negative_values_percent = negative_values / num_rows * 100

                # Min, max detection
                min_value = data[column].min()
                max_value = data[column].max()

                # Calculating range
                range_value = max_value - min_value

                # Adding mean and median calculations
                mean_value = data[column].mean()
                median_value = data[column].median()

                # Calculating quantiles
                q1_value = data[column].quantile(0.25)
                q3_value = data[column].quantile(0.75)

                # Calculating standard deviation
                std_dev = data[column].std()

                # Calculating root-mean-square deviation
                rmsd = np.sqrt((data[column] - mean_value).pow(2).mean())

                # Calculating variation
                variation = std_dev / mean_value if mean_value != 0 else None

                # Calculating kurtosis
                kurtosis = data[column].kurtosis()

                # Calculating skewness
                skewness = data[column].skew()

                # Outlier detection using Z-score
                z_scores = (data[column] - data[column].mean()
                            ) / data[column].std()
                z_outliers = data[column][np.abs(z_scores) > 3]

                # Outlier detection using IQR
                q1 = data[column].quantile(0.25)
                q3 = data[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                iqr_outliers = data[column][(data[column] < lower_bound) | (
                    data[column] > upper_bound)]
                iqr_lower_outliers = data[column][data[column] < lower_bound]
                iqr_upper_outliers = data[column][data[column] > upper_bound]

                # Outlier detection using Tukey Fences
                tf_lower_bound = q1 - 3 * iqr
                tf_upper_bound = q3 + 3 * iqr
                tf_outliers = data[column][(data[column] < tf_lower_bound) | (
                    data[column] > tf_upper_bound)]
                tf_lower_outliers = data[column][data[column] < tf_lower_bound]
                tf_upper_outliers = data[column][data[column] > tf_upper_bound]

            else:
                zero_values = None
                negative_values = None
                min_value = None
                max_value = None
                range_value = None
                mean_value = None
                median_value = None
                q1_value = None
                q3_value = None
                std_dev = None
                rmsd = None
                variation = None
                kurtosis = None
                skewness = None
                z_outliers = None
                iqr_outliers = None
                tf_outliers = None

            f.write(f"Column '{column}':\n")
            f.write(f'Data type: {column_type}\n')

            if min_value is not None and max_value is not None:
                f.write(f'Min value: {min_value}\n')
                f.write(f'Max value: {max_value}\n')

            if range_value is not None:
                f.write(f'Range: {range_value}\n')

            if mode_value is not None:
                f.write(f'Mode: {mode_value}\n')

            if mean_value is not None:
                f.write(f'Mean: {mean_value}\n')

            if median_value is not None:
                f.write(f'Median: {median_value}\n')

            if q1_value is not None:
                f.write(f'1st quartile (Q1, 0.25 quantile): {q1_value}\n')

            if q3_value is not None:
                f.write(f'3rd quartile (Q3, 0.75 quantile): {q3_value}\n')

            if std_dev is not None:
                f.write(f'Standard deviation: {std_dev}\n')

            if rmsd is not None:
                f.write(f'Root mean square deviation: {rmsd}\n')

            if variation is not None:
                f.write(f'Variation: {variation}\n')

            if kurtosis is not None:
                f.write(f'Kurtosis: {kurtosis}\n')

            if skewness is not None:
                f.write(f'Skewness: {skewness}\n')

            if z_outliers is not None and not z_outliers.empty:
                f.write(
                    f'Number of outliers detected by Z-score: {len(z_outliers)}\n')

            if iqr_outliers is not None and not iqr_outliers.empty:
                f.write(
                    f'Number of outliers detected by IQR: {len(iqr_outliers)}\n')
                f.write(
                    f'IQR lower bound: {lower_bound}, Number of values below: {len(iqr_lower_outliers)}\n')
                f.write(
                    f'IQR upper bound: {upper_bound}, Number of values above: {len(iqr_upper_outliers)}\n')

            if tf_outliers is not None and not tf_outliers.empty:
                f.write(
                    f'Number of outliers detected by Tukey Fences: {len(tf_outliers)}\n')
                f.write(
                    f'Tukey Fences lower bound: {tf_lower_bound}, Number of values below: {len(tf_lower_outliers)}\n')
                f.write(
                    f'Tukey Fences upper bound: {tf_upper_bound}, Number of values above: {len(tf_upper_outliers)}\n')

            if missing_values > 0:
                f.write(
                    f'Number of missing values: {missing_values} ({missing_values_percent:.2f}%)\n')

            if zero_values is not None and zero_values > 0:
                f.write(
                    f'Number of zero values: {zero_values} ({zero_values_percent:.2f}%)\n')

            if negative_values is not None and negative_values > 0:
                f.write(
                    f'Number of negative values: {negative_values} ({negative_values_percent:.2f}%)\n')

            f.write(f'Number of unique values: {num_unique_values}\n')

            if num_unique_values <= unique_values_threshold:
                f.write(f'Unique values of the column: {unique_values}\n')

                # Counting the number of occurrences of each unique value and calculating percentage
                value_counts = data[column].value_counts(dropna=False)
                value_counts_percent = data[column].value_counts(
                    normalize=True, dropna=False) * 100

                f.write('Counts and percentages of unique values:\n')
                for value, count in value_counts.items():
                    percent = value_counts_percent[value] if pd.notnull(
                        value) else value_counts_percent[np.nan]
                    f.write(f'{value}: {count} ({percent:.2f}%)\n')
                f.write('\n')

            else:
                # Adding an example of a non-empty value from the column
                non_empty_value = data[column].dropna().iloc[0]
                f.write(f'Example of a non-empty value: {non_empty_value}\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze a CSV file.')
    parser.add_argument('file_path', help='Path to the CSV file')
    parser.add_argument('-u', '--unique_values_threshold', type=int, default=10,
                        help='Threshold for number of unique values in a column to display (default: 10)')
    parser.add_argument('-d', '--delimiter', default=',',
                        help="Delimiter used in the CSV file (default: ',')")

    args = parser.parse_args()

    # Extracting the file name from the file path
    file_name = os.path.basename(args.file_path)
    # Removing the file extension
    file_name_no_ext = os.path.splitext(file_name)[0]
    # Creating a new file name for the analysis results
    output_file_name = file_name_no_ext + '_analysis.txt'

    # Getting the directory of the input file
    input_file_dir = os.path.dirname(args.file_path)
    # Joining the input file directory with the output file name
    output_file_path = os.path.join(input_file_dir, output_file_name)

    analyze_csv_file(file_path=args.file_path,
                     unique_values_threshold=args.unique_values_threshold,
                     delimiter=args.delimiter,
                     output_file_path=output_file_path)
