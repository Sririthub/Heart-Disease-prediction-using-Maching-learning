from flask import Flask,render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


app = Flask(__name__)


# home route page

@app.route('/')
def home():
    return render_template('home.html')




# admin login page

@app.route("/adminlogin",methods=['GET','POST'])
def adminlogin():
   admin_name = request.form['admin_name']
   admin_pass = request.form['admin_password']
   print(admin_name)
   print(admin_pass)
   if admin_name == 'admin' and admin_pass == 'admin':   
      return render_template("heart.html")
   else:
      return "something wrong........."







# 1st semester data stored in databases


@app.route('/machine_learning',methods=['GET','POST'])
def machine_learning():
    if request.method == 'POST':
        dataset = pd.read_csv("heart.csv")#reading dataset
        #print(dataset) # printing dataset

        #HUMIDITY SENSOR
        x = dataset.iloc[:,:-1].values #locating inputs
        y = dataset.iloc[:,-1].values #locating outputs

        #printing X and Y
        print("x=",x)
        print("y=",y)

        from sklearn.model_selection import train_test_split # for splitting dataset
        x_train,x_test,y_train,y_test = train_test_split(x ,y, test_size = 0.25 ,random_state = 0)
        #printing the spliited dataset
        print("x_train=",x_train)
        print("x_test=",x_test)
        print("y_train=",y_train)
        print("y_test=",y_test)




        #KNN ALGORITHM
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean' , p = 2)
        knn.fit(x_train, y_train)
        print('K NEAREST NEIGHBOUR ACCURACY LEVEL:', knn.score(x_train, y_train))
        Y_predict=knn.predict(x_test) # predicted output
        print(Y_predict)



        #RANDOM FOREST ALGORITHM
        from sklearn.ensemble import RandomForestClassifier
        classifier1=RandomForestClassifier()
        classifier1.fit(x_train,y_train)#trainig Algorithm
        print('RANDOM FOREST ACCURACY LEVEL:', classifier1.score(x_train, y_train))
        y_pred=classifier1.predict(x_test) #testing model
        print("y_pred",y_pred)






        age = request.form['age']
        sx = request.form['sx']
        chestpain = request.form['chestpain']
        restbp = request.form['restbp']
        cholesterol = request.form['cholesterol']
        fbs = request.form['fbs']
        restingecg = request.form['restingecg']
        maxheartrate = request.form['maxheartrate']
        excercise = request.form['excercise']
        oldpeak = request.form['oldpeak']
        stslope = request.form['stslope']
        print(age)
        print(sx)
        print(chestpain)
        print(restbp)
        print(cholesterol)
        print(fbs)
        print(restingecg)
        print(maxheartrate)
        print(excercise)
        print(oldpeak)
        print(stslope)

        float(age)
        float(sx)
        float(chestpain)
        float(restbp)
        float(cholesterol)
        float(fbs)
        float(restingecg)
        float(maxheartrate)
        float(excercise)
        float(oldpeak)
        float(stslope)
        
 
        output=knn.predict([[age,sx,chestpain,restbp,cholesterol,fbs,restingecg,maxheartrate,excercise,oldpeak,stslope]])
        print(output)
        if output == 0:
            a = "NO HEART DISEASE"
        else:
            a = "HEART DISEASE DETECTED"

        output1=classifier1.predict([[age,sx,chestpain,restbp,cholesterol,fbs,restingecg,maxheartrate,excercise,oldpeak,stslope]])
        print(output)
        if output1 == 0:
            B = "NO HEART DISEASE"
        else:
            B = "HEART DISEASE DETECTED"

        return render_template("output.html", OUTPUT = a , OUTPUT1 = B )

if __name__ == '__main__':
    app.run(debug=True)














