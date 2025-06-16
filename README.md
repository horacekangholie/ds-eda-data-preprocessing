### EDA - Data Preprocessing

#### Detection and Handling of Outliers
In data analysis and machine learning, outliers are observations that lie unusually far from the rest of the data—either extremely high or low. Such anomalous values can skew summary statistics or harm model performance, so it’s important to detect and then either remove or correct them during preprocessing. This section introduces several common methods for spotting outliers.

Methods for Checking Outliers
There are many techniques to identify outliers; here are a few straightforward statistical approaches:
1.  Descriptive statistics
    Compute summary measures (mean, standard deviation, quartiles, etc.) and flag values that deviate markedly—e.g. anything several standard deviations from the mean, or outside the interquartile “fences.”
2.  Visualization
    Use plots—box plots, histograms overlaid with density curves, or scatter plots—to visually inspect where data points lie relative to the bulk of the distribution.




