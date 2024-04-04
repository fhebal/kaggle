import os
import sys

import pandas as pd
from ydata_profiling import ProfileReport


def main(filepath):
    """
    Generates a minimal profile report and writes to an HTML file.

    :param filepath: Path to the input file to read into a DataFrame.
    """
    df = pd.read_csv(filepath)
    profile = ProfileReport(df, minimal=True)
    filename = os.path.basename(filepath).split(".")[0] + ".html"
    output_path = os.path.join("reports", filename)
    os.makedirs("reports", exist_ok=True)
    profile.to_file(output_path)
    print(f"Report generated and saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide a file path.")
