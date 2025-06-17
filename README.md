## 📚 EDA - Data Preprocessing

- [Detection and Handling of Outliers](#️-detection-and-handling-of-outliers)
- [Handling Missing Values](#️-handling-missing-values)
- [Handling Categorical Data](#️-handling-categorical-data)


-----------------------------

### 🗒️ Detection and Handling of Outliers

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

#### Methods for Handling Outliers

Once outliers are detected, you can either **replace** them (imputation) or **remove** them.

Each strategy has trade-offs:
-   **Imputation** preserves the record but alters its value.
-   **Removal** avoids biasing the data but discards the entire row.

Below are three common imputation rules plus row deletion.

**Median Imputation**

Use the **median** of the variable to replace any outliers. This is robust because the median itself isn't skewed by extremes.

**Mean Imputation**

Replace outliers with the **mean**. Suitable when data are symmetrically distributed---but beware that the mean itself is influenced by extremes.

**Mode Imputation**

Use the **mode** (most frequent value) to replace outliers. Works well for categorical data or numeric data with a clear peak.

**Row Removal**

If outliers are numerous or widely scattered, you may choose to remove those entire records at once. This avoids imputation bias but discards potentially useful data.

Data Cleaning and Preprocessing


### 🗒️ Handling Missing Values

Processing missing data is a critical step in data analysis and machine learning, because missing entries can undermine model performance and reliability. Much like outlier treatment in previous section, you can either **impute** missing values or **drop** them entirely, depending on the situation and the proportion of missingness. Common imputation strategies include median, mean, and mode.

```python
import pandas as pd
import numpy as np

# Build a simple height–weight dataset with two missing heights
data = {
    "Height": [145, 155, 165, 170, 175, np.nan, 180, 185, np.nan, 190],
    "Weight": [35, 45, 55, 60, 65, 70, 75, 80, 85, 90]
}

# Drop the value
df = pd.DataFrame(data)
df.dropna(inplace=True)
print("After dropping missing rows:")
print(df)

# Median Imputation
df = pd.DataFrame(data)
median_height = df['Height'].median()
df['Height'].fillna(median_height, inplace=True)

print("Median used for imputation:", median_height)
print("\nData after median imputation:")
print(df)

# Mean Imputation
df = pd.DataFrame(data)
mean_height = df['Height'].mean()
df['Height'].fillna(mean_height, inplace=True)

print("Mean used for imputation:", mean_height)
print("\nData after mean imputation:")
print(df)
```

**Other Imputation Techniques**\
Beyond median and mean, you can also consider:

-   **Nearest-neighbor imputation** (e.g. KNN-based)

-   **Predictive imputation** (build a model to predict missing values)

-   **Random-sample imputation** (draw from the distribution of observed values)

-   **Multiple imputation** (use statistical models to generate several plausible imputations)


### 🗒️ Handling Categorical Data

In many datasets you'll encounter **categorical features**---textual labels rather than numbers. Models require numeric input, so we must convert categories into numeric codes or dummy vectors. Common techniques include:

-   **Label Encoding:** map each category to a unique integer (no ordinal meaning).

-   **Ordinal Encoding:** assign integers to categories that do have a natural order.

-   **One-Hot Encoding:** create binary indicator columns for each category.

-   **Frequency Encoding:** replace categories with their occurrence counts or frequencies.

-   **Feature Combinations:** engineer new categorical features by combining existing ones.


### 🗒️ Data Normalization and Standardization

In data science and machine learning, the features of a dataset often originate from different units or scales. Such disparities can cause models to give undue weight to features with larger scales or to converge slowly during training. To eliminate these effects, we perform **normalization** and **standardization**, which make features comparable.

-   **Normalization** rescales the original data to a fixed range, typically [0, 1], without changing the overall shape of the distribution.

-   **Standardization** shifts the data so that it has zero mean and unit variance, which can improve algorithms that assume a Gaussian distribution.

**Normalization**

Normalization is a common preprocessing step whose goal is to linearly scale each feature into the [0, 1] interval. This method does not alter the shape of the distribution---only its range---making it useful when features have very different units (for example, pixel intensities in images).

A typical min--max normalization formula is:

x′=x-min⁡(x)max⁡(x)-min⁡(x)x' \;=\; \frac{x - \min(x)}{\max(x) - \min(x)}x′=max(x)-min(x)x-min(x)​

> Min--Max Normalization

Where:

-   xxx is an original data point,

-   x′x'x′ is the normalized value,

-   min⁡(x)\min(x)min(x) and max⁡(x)\max(x)max(x) are the minimum and maximum values of the feature, respectively.

Normalization ensures that all features share a common scale, which can speed up convergence for many machine-learning algorithms and prevent features with large ranges from dominating the model.



