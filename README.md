# Energy Consumption Forecasting
**Objective:** Take a combination of electrical grid and weather data to predict
energy consumption in the Michigan area.

**Data:**
- [NOAA Station monitoring](https://www.ncei.noaa.gov/pub/data/ghcn/daily/)
  - ghcnd_hcn.tar.gz and untar the folder inside with the .txt file
- [EIA Monitoring](https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/regional/REG-CENT)
  - Click the "Download Data Button" to the top right of the graphic
  - Click tab labeled "Six-Month Files"
  - Everything you need should be within the files with description BALANCE
- [EIA Consumption](https://www.eia.gov/electricity/data/browser/#/topic/2?agg=2,0,1&fuel=f&geo=00004&sec=g&linechart=ELEC.CONS_TOT.COW-MI-99.M&columnchart=ELEC.CONS_TOT.COW-MI-99.M&map=ELEC.CONS_TOT.COW-MI-99.M&freq=M&start=200101&end=202406&ctype=linechart&ltype=pin&rtype=s&pin=&rse=0&maptype=0)

**How to run**
Documents under each supervised and unsupervised have their own .yml file.
This was ran on WSL so a Linux OS is recommended for this.
To run any run the command 
1. `conda env create -f environment.yml`
2. `conda activate forecasting`
3. `jupyter notebook` 


## The `scripts` Module
**ETL**
- performs extract transform and load for the NOAA and EIA Monitoring data
- NOAA data is needed in the data folder for this to work
- Works in tandem with ETLConfig

**ETLConfig**
- Object that goes into the initialization of ETL
- Makes it a little more readable so we don't pass in 1 million parameters in `.run()`

**BayesianOptimization**
- Handcrafted bayesian optimization using sklearn as a backend
- Uses cross validated scores to get the best parameters
- Only works with the XGBoost estimator and skforecast combination

**Algorithms**
- LSTM: Trained on L4 GPU in a Google Colab
- XGBoost: Using SKforecast wrapper
- ARIMA
