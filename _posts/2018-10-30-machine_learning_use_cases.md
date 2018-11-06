---
layout: post
title:  "Data science and machine learning applications for business success"
author: weisseo
categories: [use case, business]
image: assets/images/machine_learning_use_cases/Churn.png
mathjax: true
featured: true
---

In the last years technologies such as data science, artificial intelligence
and predictive analytics have become increasingly popular.
The academic field behind these technologies is called **machine learning**.
The core idea is that the computer (the **machine**) learns autonomously from data and generates
business insights and leads to data-driven decision making.
In this article we will discuss where and how these techniques are applied successfully in industry.

## Advertising, Sales & Customer Experience

There are a lot of machine learning applications for customer data.
The key idea is to gain some insights about your customer so that you know when to contact whom,
when to offer what kind of discount, or when to suggest which product.
If other companies have similar or the same customers we could even think of insights as the product.

### Targeted Marketing

The most popular technique in advertising probably is targeted marketing.
Every individual is targeted differently so that the probability of action such as a
click on your advertisement is maximized.
In order for the machine to learn, it needs positive and negative examples in the data.
A positive example would be data from people who clicked on advertisement in the past
and negative examples are where advertisement did not work.
In online advertising click rates are recorded automatically, which makes it optimally suitable for machine learning.
But also in direct (e-mailed) advertisement it is possible to record response rates and let the machine learn from the data.

<center>
<img src="/assets/images/machine_learning_use_cases/targeted_marketing.png">
</center>

Targeted marketing is famously used by Facebook and Google.
The content of the advertisements is optimally chosen from your past click behaviour.
The advertiser has the advantage to only approach people where the effort is most effective,
and the users only get advertisements that might be interesting for them.

### Churn Prediction

Very often it can be forecast that a customer will cease their custom or switch to a competitor.
Possible predictors are decreasing use of your service, increasing complaints, or something
less obvious like using a different vocabulary in their written communication with you.
If you have enough examples of churn in your client data, machine learning can autonomously
find patterns and predict the churn event before it happens.
With these warning signs, you can try to convince the customer of remaining with you with, for example, special offers.

<center>
<img src="/assets/images/machine_learning_use_cases/Churn.png">
<br/>
<font size="-1">http://www.everythingai.co.in/2018/01/13/churn-prediction-implementation-neural-network</font>
<br/><br/>
</center>

The American delivery service FedEx used this technique to predict with an accuracy of 65% to 95% if a client will
switch to a competitor.
Another company making use of this technique is PayPal.
They use written client feedback to classify their users into high and low risk groups and are able to
predict with an 83% accuracy if a client will quit their service.

### Recommender Systems

Similar clients are interested in similar products.
This fact can be used to recommend your clients the right product.
If you have client ratings from your users, this data can be used to predict how much a
client will like a product they have not rated yet.
But even knowing only which client bought which item can be enough to recommend appropriate products.

<center>
<img src="/assets/images/machine_learning_use_cases/amazon_recommender_systems.png" height = "300">
<br/>
<font size="-1">http://www.amazon.com</font>
<br/><br/>
</center>

Famous examples of these techniques are employed by Amazon.com and Netflix.
They use their client ratings to recommend you products or movies.
At Amazon.com you have, for example, the features "Customers who bought this item also bought these items."
or "These items are frequently bought together".
Most modern eCommerce companies use these kinds of techniques nowadays.

## Product / Process Improvement and Maintenance

Machine learning is also used to make improvements to products or business processes.
With help of these technologies certain problems that previously could only be solved by humans,
can now be solved by computers and therefore make your business more efficient.
Furthermore, computers can see far more data than humans and can predict,
for example, system failure that previously could not be detected at all.

### Predictive Maintenance

The techniques that predict machinery or system failure before it has happened,
are called predictive maintenance.
Engineering companies predict from sensor data such as temperature or pressure if their machines will fail in advance.
This helps them to service their equipment before a complete, damage-induced shutdown
while scheduling maintenance only when necessary - both of which will generate cost savings.
For example, photographs of your products can be classified into the categories "damaged" and "not damaged".
Techniques involving images are often summarized under the phrase computer vision.

<center>
<img src="/assets/images/machine_learning_use_cases/predictive-mainenance.jpg" height = "300">
<br/>
<font size="-1">https://www.industrialiotseries.com/2018/07/25/predictive-maintenance</font>
<br/><br/>
</center>

To name two examples of this approach, the American railway company TTX uses predictive maintenance
to forecast the failure probability for each of hundreds of thousands of railcar wheels in
order to forecast overall annual inventory and maintenance need within a 1.5 percent margin.
Con Edison, a player in the energy sector, predicts failure of energy distribution cables,
updating risk levels that are displayed on their operators' screens three times an hour.

### Improving Support Channel Efficiency

A lot of support channels are separated into several categories and for each of them there are specialists
to respond to the specific customer requests.
To get directed to the right support channel, either the customers have to pick the right one themselves or a
member of the customer support staff needs to do so.
Classifying text into predefined categories is a classical task in machine learning and can therefore be automated.
If you have historical data where client feedback is directed to the correct support channel,
the machine can learn from that.
If the support channel is based on verbal communication, in order to be analyzed,
a transcript of the spoken language is needed.
With the help of speech recognition techniques these transcripts can be generated automatically.

Citibank uses machine learning to make their support operation more efficient.
They classify their written client feedback and direct it automatically to the right
support channel thereby making their business more efficient.

### Predicting Sales

For retailers, predicting demand for store items helps planning how much of each product
needs to be kept in the warehouse.
Store sales are influenced by many factors, including promotions, competition, school and state holidays,
seasonality, and locality. The complexity of these data and the task make it difficult to be performed by humans.

In 2015, the German retail chain Rossmann was confronted with this task.
The store managers were tasked with predicting their daily sales for up to six weeks in advance.
They outsourced their problem to the data science competition website [kaggle.com](https://www.kaggle.com/c/rossmann-store-sales).

If you are interested in how sales prediction works from a data science perspective,
you might want to check out one of our earlier articles on this subject
[here](https://exxai.github.io/2018/10/sales-prediction) or [here](https://exxai.github.io/2018/10/forecast_sales_in_retail).


### Fraud Detection and Financial Risk Management

For financial institutions it is crucial to detect fraud or classify people into risk groups.
Fraudulent customers can often be found through their abnormal behavior.
For example, when a credit card gets stolen, it may have a lot of unexpected (or abnormal in this context)
purchases charged to it.
The techniques which are used to find unusual patterns in large sets of data are called anomaly detection.

Using historical client data, it can also be calculated how likely it is for a customer to pay back a loan.
This has proven to be very helpful for financial risk management.

<center>
<img src="/assets/images/machine_learning_use_cases/fraud.png" class="center">
<br/>
<font size="-1">http://www.altosdigital.com/top-10-ecommerce-fraud-detection-prevention-tips</font>
<br/><br/>
</center>

There are several companies who use these kinds of techniques in their business.
Brasil Telecom predicts, which clients will not pay back their loan and saved four million US dollar in bad loans.
Canadian Tire manages their risk by predicting which clients are likely to pay back their loan late.

### Process Improvements in Logistics

Forecasting methods are also used in logistics in order to facilitate process improvements.
For a delivery service, it might be helpful to have a prediction of the destination addresses in advance.
This helps to plan the optimal route.
From GPS data, it can also be predicted if a courier is late and inform the receiver.

Continental Airlines is making use of these techniques to predict late flights and UPS predicts
destination addresses to optimize routing.

<center>
<img src="/assets/images/machine_learning_use_cases/logistics.jpg">
<br/>
<font size="-1">https://www.michiganstateuniversityonline.com/resources/supply-chain/logistics-fundamental-to-supply-chain-success/#.W9cbjycxnzI</font>
<br/><br/>
</center>

## References

- Eric Siegel: Predictive Analytics: The Power to Predict Who Will Click, Buy, Lie, or Die