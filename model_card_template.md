# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a supervised binary classification model based on a random forest classifier trained on the Census Income Dataset. Its purpose
is to predict whether an individual earns more than $50,000 per year based on various demographic and employment-related features.

## Intended Use
This model is intended for educational purposes and to demonstrate the application of machine learning techniques on structured data. 
It can be used to understand how demographic factors may influence income levels, 
but it should not be used for making real-world decisions regarding employment or income predictions.

## Training Data
The model was trained on the Census Income Dataset, which contains demographic and employment-related features of individuals.

## Evaluation Data
The test set used for evaluation consists of 20% of the original dataset, which was not used during training. 
This ensures that the model's performance is evaluated on unseen data.

## Metrics
The following metrics were used to evaluate the model: Precision, Recall and F1 Score. It performed as follows:

Precision: 0.739
Recall: 0.633
F1 score: 0.682


## Ethical Considerations
The following ethical considerations should be taken into account:

<ul>
   <li>Bias: The model may reflect biases present in the training data, which could lead to unfair predictions for certain demographic groups.</li>
   <li>Representation: The model's performance may vary across different demographic groups, and it is important to ensure that the training data is representative of the population.</li>
   <li>Fairness: The model was not audited for fairness, and it is crucial to assess its impact on different demographic groups before deployment.</li>
</ul>

## Caveats and Recommendations
The following caveats and recommendations should be considered:
<ul>
   <li>Small sample slices: f1 scores of 0 or 1 in categories with few samples are statistically unreliable.</li>
   <li>Data source limitations: The original dataset may contain outdated or culturally biased information, which can affect the model's predictions.</li>
   <li>Model generalizaiton: The model may not generalize well to populations that differ significantly from the training data.</li>
</ul>

