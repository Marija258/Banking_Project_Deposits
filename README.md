# Banking - Project - Deposits
---
## Project Description
Analyzing the data related to direct marketing campaigns (phone calls) of a Portuguese bank using various ML techniques. The classification goal is to predict if the client will subscribe a long-term deposit. 
## Data
The data includes personal information about each customer as well as information about the bank’s previous efforts in marketing to that client and social and economic context attributes.

## EDA
**Job status**

There are twelve types of jobs that occur in this data set. "Admin.","blue-collar"and "technician" are the most frequent.

**Education level**

There are eight education levels that occur in the data set. "University.degree", "high.school" and "basic.9y" are the most fequent.

---
![EDA](Documentation/job_education.jpg)


**Marital status**

The most frequent potential clients who are contacted are those with martial status "married", followed by "single" and "divorced".

---
![EDA](Documentation/marital.jpg)

## Evaluation Report

**Accuracy, Precison, Recall**

|Metric| DecisionTree| RandomForest|	KNN|	Gaussian|	GradientBoosting|	EXtremeGradientBoosting|
|------|-------------|--------------|----|-----------|----------------|-------------------------|
|Accuracy|	0.838454|	0.890829|	0.724633|	0.714286|	0.839867|	0.922144|
|Recall|	0.779378|	0.827970|	0.624123|	0.525299|	0.781475|	0.883763|
|Precision|	0.883800|	0.947028|	0.781150|	0.844497|	0.884806|	0.957243|

**Confusion Matrix**

![Confusion Matrix](Documentation/cm.jpg)

