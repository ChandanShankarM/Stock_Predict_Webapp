import pandas as pd # tool for data processing
from pandas_datareader import data # tool for data processing
import plotly
import plotly.graph_objects as go # tool for data visualization
import yfinance as yf # tool for downloading histrocial market data from "Yahoo! Finance"
from datetime import date # tool for manipulating dates and times
from dateutil.relativedelta import relativedelta # tool for manipulating dates and times
import numpy as np # tool for handling vectors, matrices or large multidimensional arrays
from sklearn.model_selection import train_test_split # tool for machine learning
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, render_template, Response, request, session
from plotly.offline import plot
from flask_session import Session
from pandas_datareader._utils import RemoteDataError

#Flask server side sessions
app = Flask(__name__)
SESSION_TYPE='filesystem'
app.config.from_object(__name__)
Session(app)


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/')
def index():
    return render_template('page1.html', var='Please enter the symbol of your favorite stock for analysis')


@app.route('/set_stock',methods=['POST','GET'])
def set_stock():
        form_data= request.form
        #stock=form_data['Name']
        session['stock']=form_data['Name']
        try:
            data.DataReader(session.get('stock'), 'yahoo', '2019-01-01', '2019-01-02')
        except RemoteDataError:
            return 'Sorry, Yahoo API used to extract financial data is temporarily down. Please try again later...'       
        except(KeyError, OSError):
            return render_template('page1.html', var= 'Not a valid stock symbol, try again:') 
        return render_template('page3.html', stock=session.get('stock'))#stock)

    
        
@app.route('/set_compare',methods=['POST','GET'])
def set_compare():
        form_data= request.form
        session['compare']=form_data['Name']
        try:
            data.DataReader(session.get('compare'), 'yahoo', '2019-01-01', '2019-01-02')
        except(KeyError, OSError):
            return render_template('page6.html', var= 'Not a valid stock symbol, try again:')
        fig=optB()        
        return fig       

    
def optC():
    stock=session.get('stock')
    try:
        # Save the data in "stockvariable"
        stockvariable = yf.Ticker(stock)
        # Save the earnings and revenues in "df"
        df = stockvariable.earnings
        df = df.reset_index()

        # Plotting the bar chart
        fig = go.Figure()
        # Add the revenues
        fig.add_trace(go.Bar(x=df.Year,
                    y=df.Revenue,
                    name='Revenue',
                    marker_color='rgb(55, 83, 109)'))
        # Add the earnings
        fig.add_trace(go.Bar(x=df.Year,
                    y=df.Earnings,
                    name='Earnings',
                    marker_color='rgb(26, 118, 255)'))

        # Change the layout
        fig.update_layout(
        title=f'Revenue and Earnings of {stock} Stock',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='USD (billions)',
            titlefont_size=16,
            tickfont_size=14),
        barmode='group',
        # Gap between bars of adjacent location coordinates
        bargap=0.15,
        # Gap between bars of the same location coordinate    
        bargroupgap=0.1)
        fig=plotly.offline.plot(fig, output_type='div')
        return fig
        
        
        
    # For some stocks the imported data is distorted and in a wrong format, so that an error appears
    # In this cases the user gets the following message:
    except(ValueError,AttributeError):
        return False

def optB():
    stock=session.get('stock')
    compare=session.get('compare')
    sdate = '2010-01-01'
    edate = date.today().strftime('%Y-%m-%d')
    df = data.DataReader(stock, 'yahoo', sdate, edate)
    df = df.reset_index()
    df_2 = data.DataReader(compare, 'yahoo', sdate, edate)
    df_2 = df_2.reset_index()
    fig = go.Figure()
    # Add the data from the first stock
    fig.add_trace(go.Scatter(
                x=df.Date,
                y=df['Adj Close'],
                name=f'{stock} Stock',
                line_color='deepskyblue',
                opacity=0.9))
    
    # Add the data from the second stock
    fig.add_trace(go.Scatter(
                x=df_2.Date,
                y=df_2['Adj Close'],
                name=f'{compare} Stock',
                line_color='dimgray',
                opacity=0.9))


    fig.update_layout(title=f'Price Comparison of {stock} Stock and {compare} Stock from {sdate} to {edate}', 
                      yaxis_title='Adjusted Closing Price in USD',
                     xaxis_tickfont_size=14,
                     yaxis_tickfont_size=14)
    
    fig=plotly.offline.plot(fig, output_type='div')
    return fig


def optE():
    stock=session.get('stock')
    # Save the date of today in the variable "today"
    try:
        today = date.today()
        # We convert the type of the variable in the format %Y-%m-%d
        today = today.strftime('%Y-%m-%d')
        # Save the date of today 6 months ago, by subtracting 6 months from the date of today
        six_months = date.today() - relativedelta(months=+6)
        six_months = six_months.strftime('%Y-%m-%d')
    
        df2 = yf.Ticker(stock)
        # Save the Analyst Recommendations in "rec"
        rec = df2.recommendations
        # The DataFrame "rec" has 4 columns: "Firm", "To Grade", "From Grade" and "Action"
        # The index is the date ("DatetimeIndex")

        # Now we select only those columns which have the index(date) from "six months" to "today"
        rec = rec.loc[six_months:today,]
    
        # Unfortunately in some cases no data is available, so that the DataFrame is empty. Then the user gets the following message
        if rec.empty:
            return Flase
            print(color.BOLD + color.UNDERLINE + "\n> Unfortunately, there are no recommendations by analysts provided for your chosen stock!" + color.END)
                
                
        else:    
            # Replace the index with simple sequential numbers and save the old index ("DatetimeIndex") as a variable "Date"
            rec = rec.reset_index()

            # For our analysis we don't need the variables/columns "Firm", "From Grade" and "Action", therefore we delete them
            rec.drop(['Firm', 'From Grade', 'Action'], axis=1, inplace=True)

            # We change the name of the variables/columns
            rec.columns = (['date', 'grade'])
    
            # Now we add a new variable/column "value", which we give the value 1 for each row in order to sum up the values based on the contents of "grade"
            rec['value'] = 1

            # Now we group by the content of "grade" and sum their respective values 
            rec = rec.groupby(['grade']).sum()
            # The DataFrame "rec" has now 1 variable/column which is the value, the index are the different names from the variable "grade"
            # However for the plotting we need the index as a variable 
            rec = rec.reset_index()
    
            # For the labels we assign the content/names of the variable "grade" and for the values we assign the content of "values" 
            fig = go.Figure(data=[go.Pie(labels=rec.grade,
                                            values=rec.value,
                                            hole=.3)])
            # Give a title
            fig.update_layout(title_text=f'Analyst Recommendations of {stock} Stock from {six_months} to {today}')
            fig=plotly.offline.plot(fig, output_type='div')
            return fig 
        


    # For some stocks the imported data is distorted and in a wrong format, so that an error appears
    # In this cases the user gets the following message:
    except(ValueError,AttributeError):
        return False 

        
def optD():
    stock=session.get('stock')
    try:
        stockvariable = yf.Ticker(stock)
        # Save the cashflows in "cashflow"
        cashflow = stockvariable.cashflow
        
        # Change the columnnames
        cashflow = cashflow.rename(columns=lambda x: str(x.year))

        # There are too many categories in the dataframe "cashflow"
        # We are ultimately only interested in the operative cashflow, investing cashflow, financing cashflow and the net income
        # Therefore we need to select all the columns with the respective indexes
        ICF = cashflow.loc['Total Cashflows From Investing Activities',]
        FCF = cashflow.loc['Total Cash From Financing Activities',]
        OCF = cashflow.loc['Total Cash From Operating Activities',]
        NI = cashflow.loc['Net Income',]

        # We save the variables in "CF"
        CF = [OCF, ICF, FCF, NI] 

        # With this data we create a new dataframe "cashflow"
        cashflow = pd.DataFrame(CF)
        cashflow = cashflow.reset_index()
                      
         # Change the columnnames and some values within the dataframe 
        cashflow = cashflow.rename(columns={'index': 'Cashflows'})
        cashflow.replace('Total Cash From Operating Activities', 'Cash Flow From Operating Activities', inplace=True)
        cashflow.replace('Total Cashflows From Investing Activities', 'Cash Flow From Investing Activities', inplace=True)
        cashflow.replace('Total Cash From Financing Activities', 'Cash Flow From Financing Activities', inplace=True)
        
        # Save the data of the different columns in separate variables, due to the future changes of the years or data
        first_cf = cashflow.iloc[:,1]
        second_cf = cashflow.iloc[:,2]
        third_cf = cashflow.iloc[:,3]
        fourth_cf = cashflow.iloc[:,4]


        # Plotting the bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
                x = cashflow.Cashflows,
                y = first_cf,
                name = first_cf.name))
        
        fig.add_trace(go.Bar(
                x = cashflow.Cashflows,
                y = second_cf,
                name = second_cf.name))
        
        fig.add_trace(go.Bar(
                x = cashflow.Cashflows,
                y = third_cf,
                name = third_cf.name))

        fig.add_trace(go.Bar(
                x = cashflow.Cashflows,
                y = fourth_cf,
                name = fourth_cf.name))    


        fig.update_layout(barmode = 'group',
                          bargap = 0.15,
                          bargroupgap = 0.1,
                          title = f'Cash Flow Statement of {stock} Stock',
                          xaxis_tickfont_size = 14,
                          xaxis_tickangle = -20,
                          yaxis = dict(
                                    title = 'USD (billions)',
                                    titlefont_size = 14,
                                    tickfont_size = 12))
                       
        fig=plotly.offline.plot(fig, output_type='div')
        return fig
        
        
    except(ValueError,AttributeError):
        return Flase 
    

        
def optF():
    stock=session.get('stock')
    # Get the date of today
    today = date.today()
    # Change the format
    today = today.strftime('%Y-%m-%d')

    # Get the stock data, starting from 2000-01-01 to today
    df = data.DataReader(stock, 'yahoo', '2000-01-01', 'today')
    # For the prediction we only need the column/variable "Adj Close"
    df = df[['Adj Close']]

    # Creating a variable "n" for predicting the amount of days in the future
    # We predict the stock price 30 days in the future
    n = session.get('duration')

    # Create another column "Prediction" shifted "n" units up
    df['Prediction'] = df[['Adj Close']].shift(-n)
    # We shifted the data up 30 rows, so that for every date we have the actual price ("Adj Close") and the predicted price 30 days into the future ("Prediction") 
    # Therefore the last 30 rows of the column "Prediction" will be empty or contain the value "NaN"

    # Creating independent data set "X"
    # For the independent data we dont need the column "Prediction"
    X = df.drop(['Prediction'],1)
    # Convert the data into a numpy array 
    X = np.array(X)
    # Remove the last "n" rows
    X = X[:-n]

    # Create the dependent data set "Y"
    # For the dependent data we need the column "Prediction"
    Y = df['Prediction']
    # Convert the data into a numpy array 
    Y = np.array(Y)
    # Remove the last "n" rows
    Y = Y[:-n]

    # Split the data into 80% train data and 20 % test data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

    # Create Linear Regression Model
    #lr = LinearRegression()
    lr=RandomForestRegressor(n_estimators = 300, max_depth =300, random_state = 42)
    # Train the model
    lr.fit(x_train, y_train)

    # Set "forecast" to the last 30 rows of the original data set from "Adj Close" column
    # We dont need the column 'Prediction'
    # Convert the data into a numpy array
    # We want the last 30 rows  
    forecast = np.array(df.drop(['Prediction'],1))[-n:]


    # Print the predictions for the next "n" days
    # "lr_prediction" contains the price values, which the Linear Regression Model has predicted for the next "n" (30) days
    lr_prediction = lr.predict(forecast)

    # Now we save the predictions in a DataFrame called "predictions"
    predictions = pd.DataFrame(lr_prediction, columns = ['Prediction'])
    # "predictions" has 1 column with the predicted values
    # However to plot the value we need another variable/column, which indicates the respective date

    # Therefore we replace the index of the initial data set with simple sequential numbers and save the old index ("DatetimeIndex") as a variable "Date"
    df = df.reset_index()

    # From "Date" we need the to get the last value which is the latest date and add 1 day, because that's the date when our predictions start
    d = df['Date'].iloc[-1]
    d = d + relativedelta(days =+ 1)

    # Now we make a list with the respective daterange, beginning from the startdate of our predictions and ending 30 days after
    datelist = pd.date_range(d, periods = n).tolist()
    # We add the variable to our Dataframe "predictions"
    predictions['Date'] = datelist
    # Now we have a Dataframe with our predicted values and the correspondig dates

    
    # Save the date of today 6 months ago, by subtracting 6 months from the date of today
    six_months = date.today() - relativedelta(months=+6)
    six_months = six_months.strftime('%Y-%m-%d')

    # Get the data for plotting
    df = data.DataReader(stock, 'yahoo', six_months, today)
    df = df.reset_index()

    # Plotting the chart
    fig = go.Figure()
    # Add the data from the first stock
    fig.add_trace(go.Scatter(
                    x=df.Date,
                    y=df['Adj Close'],
                    name=f'{stock} stock',
                    line_color='deepskyblue',
                    opacity=0.9))
    
    # Add the data from the predictions
    fig.add_trace(go.Scatter(
                    x=predictions.Date,
                    y=predictions['Prediction'],
                    name=f'Prediction',
                    line=dict(color='red', dash = 'dot'),
                    opacity=0.9))

    fig.update_layout(title=f'Stock Forecast of {stock} Stock for the next {n} days',
                                yaxis_title='Adjusted Closing Price in USD',
                                xaxis_tickfont_size=14,
                                yaxis_tickfont_size=14)
    
    #fig.write_image("./static/images/fig1.png")
    fig=plotly.offline.plot(fig, output_type='div')
    return fig


def optA():
    stock=session.get('stock')
    # Now the user can enter his prefered daterange (sdate = startdate; edate = enddate)
    sdate = '2010-01-01'
    edate = date.today().strftime('%Y-%m-%d')

    # We again are using the same formula as in our while True function above, because the user changed the start- and enddate
    # variable and therefore we have to redefine the variable df for plotting the user desired daterange
    df = data.DataReader(stock, 'yahoo', sdate, edate)

    # If we look at the DataFrame "df", we see that there are 6 variables/columns: "High", "Low", "Open", "Close", "Volume" and "Adj Close"
    # We also see that the Index of the "df" is the date (DatetimeIndex)
    # For the Plotting we want to use for the x-axis the date and for the y-axis the Adjusted Close values
    # However, this is only possible if both axes are assigned by variables; this is not the case with the date, because it is the index 
    # Therefore we replace the index with simple sequential numbers and save the old index ("DatetimeIndex") as a variable
    df = df.reset_index()
    # Now the DataFrame "df" has 7 variables/columns: "Date", High", "Low", "Open", "Close", "Volume" and "Adj Close"

    # Plotting the chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.Date,
                             y=df['Adj Close'],
                             line_color='deepskyblue'))
    
    # Change the axis labeling and the title
    fig.update_layout(title=f'{stock} Stock Price from {sdate} to {edate}',
                      yaxis_title='Adjusted Closing Price in USD',
                     xaxis_tickfont_size=14,
                     yaxis_tickfont_size=14)
    fig=plotly.offline.plot(fig, output_type='div')
    return fig


@app.route('/about')
def about():
    return render_template('about1.html')
    
@app.route('/home_page',methods=['POST','GET'])
def home_page():
    stock=session.get('stock')
    if request.method == 'POST':
    
        if request.form.get('stock_table') == 'Click to view top stocks and symbol':
            return render_template('page2.html')
        if request.form.get('stock_table') == 'Click to view top stocks and symbols again':
            return render_template('page7.html')    
            
        if request.form.get('back') == 'Back to main page':
            return render_template('page1.html', var= 'Please enter stock symbol:') 
        
        if request.form.get('back') == 'Back to previous page':
            return render_template('page6.html', var= 'Please enter symbol of stock to compare:')     
            
        if request.form.get('A') == 'A     Show me the price chart of my chosen stock':
            fig=optA()
            return fig
            
        if request.form.get('F') == 'F     Show me the price prediction for the future':
            return render_template('page5.html')
            
        if request.form.get('C') == 'C     Show me the revenue and earnings of my chosen stock':
            fig=optC()
            if(fig!=False):
                return fig
            else:
                return render_template('page4.html', var=f'Sorry, Revenue and Earnings of stock:{stock} is not available')
                
        if request.form.get('D') == 'D     Show me the cash flow statement of my chosen stock':
            fig=optD()
            if(fig!=False):
                return fig 
            else:
                return render_template('page4.html', var=f'Sorry, Cashflow statement of stock:{stock} is not available')
                
        if request.form.get('E') == 'E     Show me the analyst recommendations for the stock of the last 6 months':
            fig=optE()
            if(fig!=False):
                return fig 
            else:
                return render_template('page4.html', var=f'Sorry, analyst recommendations of stock:{stock} is not available')
                
        if request.form.get('B') == 'B     Show me a price comparison with an additional stock':
            return render_template('page6.html', var='Please enter the symbol of the stock to compare')
            
        if request.form.get('m1') == ' 1 Month':
            session['duration']=30
            fig=optF()
            return fig
        if request.form.get('m12') == ' 1 Year':
            session['duration']=365
            fig=optF()
            return fig
        if request.form.get('m6') == ' 6 Months':
            session['duration']=180
            fig=optF()
            return fig            

        
if __name__ == '__main__':
    app.run()            
