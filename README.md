## 📚 EDA - Data Preprocessing

- [Detection and Handling of Outliers](#️-detection-and-handling-of-outliers)

### 🗒️ Detection and Handling of Outliers
-----------------------------
In data analysis and machine learning, outliers are observations that lie unusually far from the rest of the data—either extremely high or low. Such anomalous values can skew summary statistics or harm model performance, so it’s important to detect and then either remove or correct them during preprocessing. This section introduces several common methods for spotting outliers.

#### Methods for Checking Outliers
There are many techniques to identify outliers; here are a few straightforward statistical approaches:
1.  Statistical checks

    Compute summary measures (mean, standard deviation, quartiles, etc.) and flag values that deviate markedly—e.g. anything several standard deviations from the mean, or outside the interquartile “fences.”

2.  Visualization

    Use plots—box plots, histograms overlaid with density curves, or scatter plots—to visually inspect where data points lie relative to the bulk of the distribution.

Statistical Checks
- `Mean`: The arithmetic average. Points far from the mean may be outliers.
- `Standard Deviation`: Measures spread around the mean. A large standard deviation suggests greater dispersion; points several σ away can be considered anomalous.
- `Median`: The 50th-percentile value. Less sensitive to extreme values than the mean, but still useful for comparison.
- `Quantiles`: The 25th percentile (Q1) and 75th percentile (Q3). The interquartile range (IQR = Q3 – Q1) measures the middle 50% spread. A common rule is that any point below Q1 – 1.5·IQR or above Q3 + 1.5·IQR is an outlier.

Visualization
- `Histogram`: shows the distribution of values, so you can spot extreme bars.
- `Box Plot`: displays median, quartiles and whiskers, and clearly marks outliers.
- `CDF` (Cumulative Distribution Function): plots the cumulative proportion of observations, highlighting the tails.

![Visualization Checks](/assets/visualization_checks.png)

### 🗒️ Methods for Handling Outliers
-----------------------------
Once outliers are detected, you can either **replace** them (imputation) or **remove** them.\

Each strategy has trade-offs:
-   **Imputation** preserves the record but alters its value.
-   **Removal** avoids biasing the data but discards the entire row.

Below are three common imputation rules plus row deletion.



