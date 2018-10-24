---
layout: post
title:  "Turn your raw sales numbers into deep sales insights and intelligent forecasts"
author: weisseo
categories: [forecasting]
image: assets/images/forecast_tree.png
mathjax: true
---

In the last years technologies like data science, artificial intelligence and predictive analytics have become 
increasingly popular. There are several applications of these technologies which include target marketing, churn 
prediction or sales prediction.

The academic field behind these technologies is called “machine learning”.
The core idea is that the computer (“machine”) learns autonomously from data and generates business insights and leads 
to data-driven decision making.

To illustrate and demystify the technology, in this article we will walk through the example of sales prediction. We 
will cover how to get from raw data to valuable predictions with machine learning. The article can be understood by 
nontechnical readers.

On the platform “Kaggle.com” companies can upload their data and let data scientist compete to build the best machine 
learning algorithm for their problem. One of the largest Russian software firms, 1C Company, provided their data for
competition. The goal of this competition is to predict the sales for the next month. 
Predicting the sales of store items helps planning how much products have to be kept in the warehouse.
Store sales are influenced by
many factors, including promotions, competition, school and state holidays, seasonality, and locality. And if it is 
about predicting from data, machine learning is definitely the right technology.

We have 33 months of sales data and want to predict the sales for the next month. Let us have a look at what we want to
predict:

<table class="dataframe" border="1">
  <thead>
    <tr style="text-align: right;">
      <th>ID</th>
      <th>shop_id</th>
      <th>item_id</th>
      <th>item_cnt_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>5</td>
      <td>5037</td>
      <td>?</td>
    </tr>
    <tr>
      <td>1</td>
      <td>5</td>
      <td>5320</td>
      <td>?</td>
    </tr>
    <tr>
      <td>2</td>
      <td>5</td>
      <td>5233</td>
      <td>?</td>
    </tr>
    <tr>
      <td>3</td>
      <td>5</td>
      <td>5232</td>
      <td>?</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>5268</td>
      <td>?</td>
    </tr>
  </tbody>
</table>

Given the shop id and a particular item, we want to predict how many items will be sold in the month (item count month). 
Let us now have a look at a 
snippet of how our past sales data looks like:

<table class="dataframe" border="1">
  <thead>
    <tr style="text-align: right;">
      <th>date</th>
      <th>date_block_num</th>
      <th>shop_id</th>
      <th>item_id</th>
      <th>item_price</th>
      <th>item_cnt_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>16.06.2013</td>
      <td>5</td>
      <td>30</td>
      <td>11496</td>
      <td>399.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>14.06.2013</td>
      <td>5</td>
      <td>30</td>
      <td>11244</td>
      <td>149.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>06.06.2013</td>
      <td>5</td>
      <td>30</td>
      <td>11388</td>
      <td>898.85</td>
      <td>2.0</td>
    </tr>
    <tr>
      <td>15.06.2013</td>
      <td>5</td>
      <td>30</td>
      <td>11249</td>
      <td>399.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>13.06.2013</td>
      <td>5</td>
      <td>30</td>
      <td>8081</td>
      <td>299.00</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>

Here, one row in the table is one transaction on a particular day. We also have the additional columns “date_block_num” 
which enumerates the months and “item price”. “item_cnt_day” stands for how many items have been sold with this 
particular transaction. It can also be a negative value if an item was returned.
At the moment, the data is not yet in the form, we want it to have. We want to predict the monthly sales for a
particular item and shop. So, we have to count all the items that have been sold in a month and sum up the 
“item_cnt_day”. In data science jargon these transformations are called pivot tables or “groupbys”. It is part of the 
“data wrangling process”: Transforming the raw data into the desired form. This part can take most of the data 
scientists time. After the transformation our data table looks like that:

<table class="dataframe" border="1">
  <thead>
    <tr style="text-align: right;">
      <th>date_block_num</th>
      <th>shop_id</th>
      <th>item_id</th>
      <th>year</th>
      <th>month</th>
      <th>item_cnt_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>5</td>
      <td>2</td>
      <td>30</td>
      <td>2013</td>
      <td>6</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2</td>
      <td>482</td>
      <td>2013</td>
      <td>6</td>
      <td>2.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2</td>
      <td>491</td>
      <td>2013</td>
      <td>6</td>
      <td>2.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2</td>
      <td>835</td>
      <td>2013</td>
      <td>6</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2</td>
      <td>839</td>
      <td>2013</td>
      <td>6</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>

For example, in June 2013, the item “482” was sold 2 times in shop 2. Let us now try to predict the sales of the next
month without any machine learning. We will use the sales of the last month to predict the sales of this month. We have 
to add the column “item_cnt_last_month” for that. So, for the row June 2013, shop 2, item “482”, we have to search for 
the entry May 2013, shop 2, item “482” and copy the data from “item_cnt_mont”. This is another example of 
“data wrangling”. Now our table looks like this:

<table class="dataframe" border="1">
  <thead>
    <tr style="text-align: right;">
      <th>date_block_num</th>
      <th>shop_id</th>
      <th>item_id</th>
      <th>year</th>
      <th>month</th>
      <th>item_cnt_month</th>
      <th>item_cnt_last_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>5</td>
      <td>2</td>
      <td>30</td>
      <td>2013</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2</td>
      <td>482</td>
      <td>2013</td>
      <td>6</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2</td>
      <td>491</td>
      <td>2013</td>
      <td>6</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2</td>
      <td>835</td>
      <td>2013</td>
      <td>6</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2</td>
      <td>839</td>
      <td>2013</td>
      <td>6</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>

Now we have the variable “item_cnt_month” that we want to predict in one column and our prediction “item_cnt_last_month”
in the column next to it. We can see that our prediction is not 100% accurate, but we are also not that far off. How do 
we evaluate how good our predictions are? We have to calculate the so-called scoring or loss function. One possibility 
would be to just calculate the difference between our prediction and the true sales and take the mean. For our five data
points, we are four times off by 1 and one time off by 2. So,

$$L = \frac{4 \cdot 1 + 2}{5} = 1.2$$

In practice we use the RMSE (root mean squared error). Instead of using the absolute deviation as above, we square the 
deviation and take the square root at the end. This approach penalizes large deviations more. In our example this gives 
us

$$RMSE = \sqrt{\frac{4 \cdot 1^2 + 2^2}{5}} \approx 1.26$$

Now we know how good our predictions are. The lower the RMSE, the better. There are some commonly used error functions 
(also called “loss function”) like RMSE, but in practice it can be anything. For example, it could be how much money you
make if you can make a connection between the data science problem and your finances. In the data science practice, the 
error function is tailored to the specific use case. In the example above, we used only five data points. For the 
complete last month that we have the true sales data for, we obtain a RMSE of

This was all done with simple data manipulation. No machine learning was included so far. Now let us get to the next 
step. We will not only use the item counts from last month but also from two months before that. We do a similar data 
transformation as above and have these two additional features in our data table:

<table class="dataframe" border="1">
  <thead>
    <tr style="text-align: right;">
      <th>date_block_num</th>
      <th>shop_id</th>
      <th>item_id</th>
      <th>year</th>
      <th>month</th>
      <th>item_cnt_month</th>
      <th>item_cnt_last_month</th>
      <th>item_cnt_two_months_ago</th>
      <th>item_cnt_three_months_ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>5</td>
      <td>2</td>
      <td>30</td>
      <td>2013</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2</td>
      <td>482</td>
      <td>2013</td>
      <td>6</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2</td>
      <td>491</td>
      <td>2013</td>
      <td>6</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2</td>
      <td>835</td>
      <td>2013</td>
      <td>6</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2</td>
      <td>839</td>
      <td>2013</td>
      <td>6</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>

We could now take the mean of the item counts of the last three months as a prediction. But is the last month sales not 
more important than the sales three months ago? But how much more important? And could we not use the information from 
the features “year” and “month”? Maybe there is some seasonality effect and people buy more products in summer. Here is 
the good news: machine learning will do all that for us! It will learn from the data how to use the information in the 
features.

There are several different machine learning methods like support vector machines, neural networks or decision trees. 
Here, we will introduce decision trees because they are very intuitive and easy to understand.

![Decsion Tree](assets/images/forecast_tree.png)

Have a look at the decision tree that was created of one month of training data. Let us take the first row in our table 
and see what our prediction would be. The item count last month was 0. Since this is smaller than 2.5, we take the left 
path down. The item count two months ago was also 0. Again, we go down the left path and we predict the value 0.3. This 
is a very simple decision tree with only two levels. In practice the trees are a lot larger. If we train a larger 
decision tree, we obtain a RMSE of . This is better than just predicting the last month’s item count.

The decision tree is created in that way that the error function (RMSE) is minimized. It is important to note that we 
cannot test our model on the same data that our decision tree was trained on. Our decision tree was created with data 
from month 1-4 in that way that the RMSE is small on these particular months. The RMSE error that really counts is the 
RMSE on month 5 that the algorithm has not seen yet. Otherwise, we could build an extremely large tree that would 
perfectly predict the sales on month 1-4 but would be useless on month 5 because in interpreted too much into the data 
of month 1-4. In machine learning, this is called “overfitting”. It is comparable to memorizing all the answers of last 
year’s exam in university or high school. If this year’s exam is the same as last year’s than your answers would be 
perfect. But this is rarely the case in reality. The exams differ slightly from year to year, so it is better to really 
understand the material in order to generalize well.

Decision trees are very intuitive and also has the advantage that we can get an insight into which feature is the most
important. The higher up the feature is, the more important it is. In our case, this means that the item count last 
month is more important than the item count two months ago. Not all machine learning algorithms are that easy to 
interpret. Fortunately, there are other methods which show us how important a feature is. In the next table you see the
so-called permutation importance. It shows us how important a feature is for our prediction. As expected, the sales from
last months are the most important feature. The item counts from the months before that and the item id and shop id 
also play a role. Surprisingly, the year and month are irrelevant for our prediction. Seasonality seems to not have any
influence on the sales. That is also a valuable insight.
  
![Permutation Importance](assets/images/permutation_importance.png)

Now that we built and tested the model on our training data, we can predict the unknown column from our first table:

<table class="dataframe" border="1">
  <thead>
    <tr style="text-align: right;">
      <th>ID</th>
      <th>shop_id</th>
      <th>item_id</th>
      <th>item_cnt_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0.0</td>
      <td>5</td>
      <td>5037</td>
      <td>1.200000</td>
    </tr>
    <tr>
      <td>1.0</td>
      <td>5</td>
      <td>5320</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>2.0</td>
      <td>5</td>
      <td>5233</td>
      <td>1.336957</td>
    </tr>
    <tr>
      <td>3.0</td>
      <td>5</td>
      <td>5232</td>
      <td>0.960000</td>
    </tr>
    <tr>
      <td>4.0</td>
      <td>5</td>
      <td>5268</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>

This is, of course, a simplified version in comparison with a real data science project. It is possible to have a lot
more features in the data that can be used by the algorithm. 
We could also try to use information outside the data set like how many state holidays the given month had or 
prevailing weather conditions at the time.