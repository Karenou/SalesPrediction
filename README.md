Sales Prediction
====
Data Input
----------
11,764 products sales and session data from 2019-04-01 to 2020-11-30 <br>
Features include: 
* sales data 
	*sales, quantity, standard_price, discount_price, multibuy_unit_price, promotion_type, has_freebie, freebie_price
* product list
* holiday 
* session data 
	*click, addTocart, purchase, impression, checkout
* vendor_name
	* used for product segmentation

Use one- or two-week data input to predict the week after next week sales quantity, leave one-week due to time for inventory replenishment <br>
* timestep refers to number of week data input

Product Segmentation
----------
Due to long-tail effect in E-commerce, perform product segmentation based on time series features of sales pattern 
* group 0: 6,809 long-tail items with  daily sales below 80, with many 0 daily sales
* group 1: 3,077 long-tail items with  daily sales below 80, more volatile than group 0 
* group 2: 1,383 long-tail items with more 0 daily sales than group 0
* group 3: 195 hot items with  daily sales between 0 and 600 

Model Training
---------
Boosting - LightGBM <br>
RNN - LSTM <br>

Requirements
---------
Install tensorflow, keras, tabulate, tsfresh packages