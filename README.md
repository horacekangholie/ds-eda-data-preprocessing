## 📚 EDA - Data Preprocessing

- [Detection and Handling of Outliers](#️-detection-and-handling-of-outliers)
- [Handling Missing Values](#️-handling-missing-values)
- [Handling Categorical Data](#️-handling-categorical-data)
- [Normalization and Standardization](#️-data-normalization-and-standardization)
- [Feature Scaling and Transformation](#️-feature-scaling-and-transformation)


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

In data science and machine learning, features often come in different units and scales. These discrepancies can cause some features to dominate others during model training or slow down convergence. To eliminate such issues, we apply **normalization** and **standardization**, making all features comparable.

-   **Normalization** rescales each feature into a fixed range (usually [0, 1]) without changing its distribution's shape.

-   **Standardization** shifts and scales data so it has zero mean and unit variance, which many algorithms assume.


**Normalization**

Normalization (also called min--max scaling) linearly transforms a feature so that its minimum becomes 0 and its maximum becomes 1:

> Equation 1 Min–Max Normalization

$$
x′ = \frac{x - \min(x)}{\max(x) - \min(x)}
$$

- $x$ is the original value.

- $x′$ is the normalized value.

- $min⁡(x)$ and $max⁡(x)$ are the feature's minimum and maximum, respectively.

This technique preserves the shape of the original distribution but brings all features to a common scale, which helps many machine-learning algorithms converge faster and treat each feature equally.

**Standardization**

Standardization is another key preprocessing technique whose goal is to transform each feature so that it has zero mean and unit variance—that is, to follow a standard normal distribution N(0,1)N(0,1). After standardization, the data are more likely to satisfy the normality assumptions of algorithms such as linear regression, logistic regression, and support vector machines. Standardization also helps reduce the influence of outliers by putting all features on the same scale, so the model treats each feature equally.

A common “z-score” standardization formula is:

> Equation 2 Mean & variance (z-score) standardization

$$
x' = \frac{x−μ}{σ}
$$

- $x$ is an original data point

- $x′$ is the standardized value

- $μ$ is the mean of the feature

- $σ$ is the standard deviation of the feature

Standardization (also called z-score normalization) converts features into a common unit-variance format. It’s especially useful for algorithms that assume normally distributed inputs or that are sensitive to the relative scales of features. By standardizing, you often speed up convergence in gradient-based methods and improve numerical stability.

**Why Do We Need Feature Scaling?**

Many machine-learning algorithms rely on distance or gradient calculations and are therefore sensitive to the scale of input features. For example:

-   **K-Nearest Neighbors (KNN)** computes distances directly in feature space---unequal scales will bias its notion of "nearest."

-   **Support Vector Machines (SVMs)** seek to find maximum-margin hyperplanes---standardizing inputs often yields better separators.

-   **K-Means Clustering** uses Euclidean distances to assign points to clusters, so features must share a common scale.

Proper feature scaling and transformation help:

1.  **Speed up model convergence**
    When features lie on similar scales, optimization (e.g. gradient descent) converges more quickly.

2.  **Improve model accuracy**
    By removing scale disparities, you avoid having the model over-rely on features with large magnitudes, boosting overall predictive performance.

3.  **Reduce the impact of outliers**
    Standardized data tend to dampen the effect of extreme values so that they don't unduly dominate model fitting.

##### Algorithms that Require Scaling

**Distance based algorithms** - Requires scaling

- **K-Nearest Neighbors (KNN):** Distance-based, so needs consistent scales.

- **Support Vector Machine (SVM):** Sensitive to feature ranges for finding optimal margins.

- **K-Means Clustering:** Based on Euclidean distance, assumes features share the same units.

Scaling (normalization or standardization) is thus a critical step both during data preprocessing and often again before model training to ensure stable, efficient, and accurate learning.

**Gradient-Descent based algorithms**

- **Linear Regression**: Standardizing features can speed up the convergence of gradient-descent optimization.

- **Logistic Regression**: Likewise benefits from feature scaling---gradient descent converges more quickly.

- **Neural Networks**: Standardized inputs stabilize training and help the network converge faster.

**Dimensionality-Reduction algorithms**

- **Principal Component Analysis (PCA)**: A widely used linear method that projects data onto orthogonal "principal components." Before applying PCA, you should standardize features so that each has the same scale---otherwise large-scale features dominate the component calculations.

- **t-Distributed Stochastic Neighbor Embedding (t-SNE)**: A non-linear technique for mapping high-dimensional data into 2D or 3D for visualization. You should first normalize or standardize your data to keep the values in a reasonable range.

- **Linear Discriminant Analysis (LDA)**: Finds linear combinations that best separate multiple classes. LDA assumes normally distributed features with equal class covariances---so standardization or normalization is typically required beforehand.

##### Algorithms That Do Not Require Scaling

**Tree-based Methods**

- **Decision Trees**: Splits depend on thresholding single features, so they are insensitive to feature scale.

- **Random Forests**: An ensemble of decision trees---also scale-invariant.

- **Gradient Boosting Machines** (GBM; e.g. XGBoost, LightGBM, CatBoost): Since they build decision trees, they too do not require feature scaling.


### 🗒️ Feature Scaling and Transformation

In this section, we’ll use the Iris dataset as a running example to demonstrate how to apply various preprocessing methods in Python. The sklearn.preprocessing module provides tools for normalization, standardization, and more. We’ll show how to load the Iris data, apply each scaler, and prepare the features for machine-learning models.

##### MinMaxScaler

MinMaxScaler rescales each feature into a given range (default is [0, 1]). It maps the minimum of each feature to 0 and its maximum to 1, with a linear transformation for all intermediate values.

##### MaxAbs Scaler (Maximum Absolute Value Scaling)

MaxAbsScaler scales each feature by its maximum absolute value, so that after transformation all values lie in [−1,1][−1,1]. Unlike min–max scaling, it does not shift data—only scales by the feature’s largest magnitude.

#### Standard Scaler (Mean & Variance Standardization)

StandardScaler standardizes each feature by removing its mean and scaling to unit variance, so that the resulting distribution has mean 0 and standard deviation 1. This is especially useful when features are roughly Gaussian or when models assume centered inputs.

#### Robust Scaler (Median & IQR Standardization)

The RobustScaler scales each feature so that its median becomes 0 and its interquartile range (IQR = Q3 – Q1) becomes 1. Because it uses the median and IQR—both robust statistics—it is far less sensitive to outliers than standard z-score scaling.


#### fit_transform() VS transform() 

- `fit_transform()` - on training data (learns and applies parameters)

- `transform()` - on test/new data (applies the same parameters)
