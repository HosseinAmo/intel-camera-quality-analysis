"""
analyze_intel_quality.py

This script reads the CSV created by build_intel_quality_dataset.py
('intel_camera_quality.csv'), applies simple pass/fail rules based on
brightness and contrast, computes statistics, and plots distributions.

Run this AFTER running build_intel_quality_dataset.py.
"""

import pandas as pd          # For loading and analyzing the CSV data
import matplotlib.pyplot as plt  # For plotting charts


# -----------------------------
# MAIN SCRIPT LOGIC
# -----------------------------

def main():
    """
    Main function that:
      1. Loads intel_camera_quality.csv
      2. Shows basic statistics for brightness and contrast
      3. Applies simple specification limits to classify images as PASS/FAIL
      4. Prints summary statistics and failure breakdown
      5. Plots brightness and contrast distributions and brightness by label
      6. Saves an annotated CSV with pass/fail status
    """

    # Name of the CSV file that should already exist in the same folder
    input_csv = "intel_camera_quality.csv"

    # Load the CSV into a DataFrame
    print(f"Loading data from {input_csv} ...")
    df = pd.read_csv(input_csv)

    # Show the first few rows to confirm the structure
    print("First few rows of data:")
    print(df.head())
    print()

    # -------------------------
    # Basic Descriptive Statistics
    # -------------------------

    # Show some statistics about brightness (min, max, mean, etc.)
    print("Brightness statistics:")
    print(df["Brightness"].describe())
    print()

    # Show some statistics about contrast
    print("Contrast statistics:")
    print(df["Contrast"].describe())
    print()

    # -------------------------
    # Specification Limits (Fake, but plausible)
    # -------------------------

    # These thresholds are arbitrary, but based on grayscale range [0, 255]:
    # If brightness < BRIGHTNESS_MIN ⇒ the image is considered "too dark"
    # If brightness > BRIGHTNESS_MAX ⇒ the image is "too bright / washed out"
    # If contrast   < CONTRAST_MIN   ⇒ the image is "low contrast / flat"
    BRIGHTNESS_MIN = 60
    BRIGHTNESS_MAX = 200
    CONTRAST_MIN = 20

    def classify_row(row):
        """
        Decide if a single image passes or fails based on brightness & contrast.

        Parameters
        ----------
        row : pandas.Series
            A row from the DataFrame, containing Brightness and Contrast.

        Returns
        -------
        (status, reasons) : (str, str)
            status  = "PASS" or "FAIL"
            reasons = "" if PASS, or a string like "too_dark;low_contrast"
        """
        # We collect all failure reasons in this list
        reasons = []

        # Check each condition and add a reason if triggered
        if row["Brightness"] < BRIGHTNESS_MIN:
            reasons.append("too_dark")

        if row["Brightness"] > BRIGHTNESS_MAX:
            reasons.append("too_bright")

        if row["Contrast"] < CONTRAST_MIN:
            reasons.append("low_contrast")

        # If we have any reasons, it's a FAIL; otherwise, it's a PASS
        if reasons:
            status = "FAIL"
            # Join multiple reasons with semicolons, e.g. "too_dark;low_contrast"
            reason_str = ";".join(reasons)
        else:
            status = "PASS"
            reason_str = ""

        return status, reason_str

    # Apply the classify_row function to each row.
    # result_type="expand" tells pandas to split the returned tuple into 2 columns.
    statuses = df.apply(classify_row, axis=1, result_type="expand")

    # The first element (index 0) is the status ("PASS" or "FAIL")
    df["Status"] = statuses[0]

    # The second element (index 1) is the concatenated reasons string
    df["Fail_Reasons"] = statuses[1]

    # -------------------------
    # Overall Pass/Fail Summary
    # -------------------------

    total = len(df)                          # Total number of images
    failed = (df["Status"] == "FAIL").sum()  # Number of failed images
    passed = total - failed                  # Number of passed images

    # Avoid division by zero, but total should not be zero anyway
    failure_rate = (failed / total * 100) if total > 0 else 0.0

    print(f"Total images: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Failure rate: {failure_rate:.2f}%")
    print()

    # -------------------------
    # Failure Reasons Breakdown
    # -------------------------

    # Filter only the failed images
    fail_only = df[df["Status"] == "FAIL"]

    if not fail_only.empty:
        # Count how many times each reason combination appears
        reason_counts = fail_only["Fail_Reasons"].value_counts()
        print("Failure reasons breakdown (combinations):")
        print(reason_counts)
        print()
    else:
        print("No failed images, so no failure reasons to show.")
        print()

    # -------------------------
    # Pass/Fail by Label (Scene Type)
    # -------------------------

    # Group by Label and Status and count occurrences
    label_status_counts = df.groupby(["Label", "Status"]).size().unstack(fill_value=0)

    print("Pass/Fail by scene type (label):")
    print(label_status_counts)
    print()

    # -------------------------
    # Plot: Brightness Distribution
    # -------------------------

    plt.hist(df["Brightness"], bins=30)
    plt.title("Brightness Distribution")
    plt.xlabel("Brightness")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Plot: Contrast Distribution
    # -------------------------

    plt.hist(df["Contrast"], bins=30)
    plt.title("Contrast Distribution")
    plt.xlabel("Contrast")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Plot: Boxplot of Brightness by Label
    # -------------------------

    # This shows how brightness varies across scene types (buildings, forest, etc.)
    df.boxplot(column="Brightness", by="Label", rot=45)
    plt.title("Brightness by Scene Type")
    # Remove the automatic "Boxplot grouped by Label" super-title
    plt.suptitle("")
    plt.xlabel("Scene Type")
    plt.ylabel("Brightness")
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Save Annotated CSV
    # -------------------------

    # Save the DataFrame with Status and Fail_Reasons columns added
    output_csv = "intel_camera_quality_annotated.csv"
    df.to_csv(output_csv, index=False)
    print(f"Annotated data saved to {output_csv}")


# This ensures main() runs when we call "python analyze_intel_quality.py"
if __name__ == "__main__":
    main()
