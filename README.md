# cosc5p70

A predictive model for academic advisors to predict student performance in higher education early. To understand and address potential student issues at their first signs.

## Data sources
Predict Students' Dropout and Academic Success<sup>1</sup> is a dataset created by Realinho et. al. (2021). It consists of information of students enrolled in various undergraduate programs such as agronomy, design, education, nursing, journalism, management, social service, and technologies. The dataset includes students' academic performance at the end of the first and second semesters<sup>1</sup>. It includes a target label to denote whether each student successfully _graduated_ from the program, _dropped out_, or is still currently _enrolled_.

We use this dataset to train our model to predict how likely a student will drop out by using the various demographic and socio-economic information present in the data. We train a black-box model (a feed-forward neural network) on the data and present a user-friendly interface for academic advisors to make inferences from student data.

## Preprocessing methods
We apply a simple standardization procedure to all the input features used. The procedure is as follows: $\frac{x - \hat{x}}{\sigma}$, where $\hat{x}$ is the mean and $\sigma$ is the standard deviation for each feature in the dataset. This ensures that data set has a mean of 0 and a standard deviation of 1.

## Ethical concerns
An ethical concern that arose was the use of sensitive demographic data that could potentially lead to bias: e.g., gender and nationality. To this effect, we decided to remove these two features from the model training to prevent gender and ethnicity bias in the model.

---
[1] Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). Predict Students' Dropout and Academic Success [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89.

[2] M.V.Martins, D. Tolledo, J. Machado, L. M.T. Baptista, V.Realinho. (2021) "Early prediction of student’s performance in higher education: a case study" Trends and Applications in Information Systems and Technologies, vol.1, in Advances in Intelligent Systems and Computing series. Springer. DOI: 10.1007/978-3-030-72657-7_16
