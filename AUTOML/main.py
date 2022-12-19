from flask import Flask, render_template,request,send_from_directory
import pandas as pd
app = Flask(__name__)

# importing pandas as pd
import pandas as pd
from pandas_profiling import ProfileReport,profile_report
import time
#import pandas-profiling
import os

#import sweetviz
import csv
from IPython.display import HTML

# creating the dataframe


#print("Original DataFrame :")
#display(df)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/next',methods=['get','post'])
def next():
    #print(request.files['file'])
    data1=[]
    f = request.files['filename']

    # data = pd.DataFrame(f)
    # print(data)
    for i in f:
        output=(str(i,'UTF-8'))
        data1.append(output)

    x = []
    for i in data1:
        if '\r\n' in i:
            data = i
            x.append(data[:-2].split(","))
        else:
            x.append(i.split(","))

    df = pd.DataFrame(x)
    new_header = df.iloc[0]  # grab the first row for the header
    df = df[1:]  # take the data less the header row
    df.columns = new_header  # set the header row as the df header

    data1=df.copy()
    data1.to_csv('data1.csv',index=False)
    columns_name=new_header

    # data1 = pd.DataFrame(data1)
    # x=[]
    # for i in data1[0]:
    #     x.append(i)
    # data1=pd.DataFrame(x)

    #result = data.to_html()
    return render_template('index2.html', data1=data1.to_html(), data=columns_name)
   # return user_input

@app.route("/test" , methods=['GET', 'POST'])
def test():
    #select = request.form.get('comp_select')
    data = pd.read_csv('data1.csv')
    profile=ProfileReport(data)
    profile.to_file('templates/output.html')
    time.sleep(3)
    return render_template("output.html") # just to see what select is

#####################
@app.route("/test2" , methods=['GET', 'POST'])

def test2():
    from pycaret.regression import setup,pull,compare_models,save_model

    select = request.form.get('comp_select')
    data = pd.read_csv('data1.csv')
    setup(data,target=select,silent=True)
    setup_df=pull()
    best_model = compare_models()
    compare_df=pull()
    final_df = pd.DataFrame(compare_df)

    #profile=ProfileReport(data)
    #profile.to_file('templates/output.html')
    #time.sleep(3)
    return render_template("index4.html",data=final_df.to_html(),best_model=best_model) # just to see what select is

@app.route('/Regressiontarget',methods=['GET', 'POST'])
def Regressiontarget():
    data=pd.read_csv('data1.csv')
    columns = data.columns
    return render_template("index6.html",data=columns)

@app.route('/Classificationtarget',methods=['GET', 'POST'])
def Classificationtarget():
    data=pd.read_csv('data1.csv')
    columns = data.columns
    return render_template("index7.html",data=columns)

@app.route('/Regression', methods=['GET', 'POST'])
def testcase1():
    from pycaret.regression import setup, pull, compare_models, save_model

    select = request.form.get('comp_select')
    data = pd.read_csv('data1.csv')
    setup(data, target=select, silent=True)
    setup_df = pull()
    best_model = compare_models()
    compare_df = pull()
    final_df = pd.DataFrame(compare_df)
    save_model(best_model,"templates/best_model")

    # profile=ProfileReport(data)
    # profile.to_file('templates/output.html')
    # time.sleep(3)
    return render_template("index4.html", data=final_df.to_html(), best_model=best_model)

@app.route('/Classification', methods=['GET', 'POST'])
def testcase2():
    from pycaret.classification import setup, pull, compare_models, save_model

    select = request.form.get('comp_select')
    data = pd.read_csv('data1.csv')
    setup(data, target=select, silent=True)
    setup_df = pull()
    best_model = compare_models()
    compare_df = pull()
    final_df = pd.DataFrame(compare_df)
    save_model(best_model, 'templates/best_model')

    # profile=ProfileReport(data)
    # profile.to_file('templates/output.html')
    # time.sleep(3)
    return render_template("index4.html", data=final_df.to_html(), best_model=best_model)


@app.route('/download-pickle')
def download_pickle():
  return send_from_directory('templates', 'best_model.pkl', as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)