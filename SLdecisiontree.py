import pandas as pd, numpy as np, time 
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import graphviz
import time
# YongBaek Cho


def get_frame():
    #DataFrame from cars_and_temps_F18.csv
    #Return the frame 
    df = pd.read_csv('cars_and_temps_F18.csv', usecols=[0,1,2,3,4])
    df = df.dropna(subset=['other_T'], how = 'all')
    df = df.reset_index(drop = True)
    
    return df

def clean_frame(df):
    #This function takes a frame as created by the get_dataframe() function and cleans it in place
    coulmn = []
    for col in df.columns:
        coulmn.append(col.strip())
    df.columns = coulmn

    for row in df.index:
        if df.loc[row,'color'].lower() == 'silver' or df.loc[row,'color'].lower() == 'gold' or 'light' in df.loc[row,'color'].lower():
            df.loc[row,'color'] = 'light'
        elif df.loc[row,'color'].lower() == 'white' or df.loc[row,'color'].lower() == 'black' or df.loc[row,'color'].lower() == 'light' or df.loc[row,'color'].lower() == 'dark':
            df.loc[row,'color'] = df.loc[row,'color'].strip().lower()
        else:
            df.loc[row,'color'] = 'dark' 

    for row in df.index:
        if 'n' in df.loc[row,'shade'].lower():
            df.loc[row,'shade'] = 'n'
        elif 'y' in df.loc[row,'shade'].lower():
            df.loc[row,'shade'] = 'y'
        else:
            df.loc[row,'shade'] = 'p'

    for row in df.index:
        if df.loc[row,'finish'] == 'gloss' or df.loc[row,'finish'] == 'Glossy':
            df.loc[row,'finish'] = 'glossy'
        elif pd.isnull(df.loc[row,'finish']):
            df.loc[row,'finish'] = 'unk'
        else:
            df.loc[row,'finish'] = df.loc[row,'finish'].lower()

def get_X_and_y(df):
    # this function takes a cleaned frame and returns a DataFrame of our training data and a vector (Series) of our labels.
    y = []
    X = df[['hood_T','color','shade','finish']]
    y = pd.Series(df['other_T'].values)

    return X,y

def bin_y(v,numBins = 5):
    #  this function takes a labels vector and a number of bins with a default value of 5 and returns a vector of the same length that contains bin numbers instead of labels.  
    width = 100 / numBins

    for i in range(len(v)):
        if v[i] <= 90 + width:
            v[i] = 0
        elif 90 + width < v[i] <= 90 + (width*2):
            v[i] = 1
        elif 90 + (width*2)< v[i] <= 90 + (width*3):
            v[i] = 2
        elif 90 + (width*3) < v[i] <= 90 + (width*4):
            v[i] = 3
        elif 90 + (width*4) < v[i] <= 90 + (width*5):
            v[i] = 4

    return v
def make_and_plot_tree(df, v, model_name, depth = None, sequence = None, file_sans = "temp", view = True):
    #this function takes a training data frame, a labels vector, the name of a model (choices are "DecisionTreeClassifier", 
    #"DecisionTreeRegressor", and "RandomForestRegressor"), maximum tree depth with a default value of None, a sequence of class names with a default value of None, a filename sans extension with a default value of "temp"., and a parameter view=True
    if model_name == 'DecisionTreeClassifier':
        model = DecisionTreeClassifier(max_depth = depth)
        
     
    elif model_name == 'RandomForestRegressor':
        model = RandomForestRegressor(max_depth= depth)
        model.fit(df,v)
        return model
    else:
        model = DecisionTreeRegressor(max_depth= depth)
    model.fit(df,v)    
    
    dotfile = export_graphviz(model, out_file = None, rounded = True, filled = True, feature_names = df.columns)
    graph = graphviz.Source(dotfile, format = "png")
    graph.render(file_sans, view = view)
    
    return model
def make_and_test(df, v, model_name, depth = None, sequence = None, file_sans = "temp", view = True):
    #this function takes the same arguments as the previous function
    model = make_and_plot_tree(df,v,model_name,depth, sequence = None, file_sans = "temp", view = True)
    d = str(depth) if depth else 'inf'
    maxdep = 'Tree accuracy ' + model_name + ', all data max depth = ' + d + ': ' + str(model.score(df,v))
    crss_val = cross_val_score(model,df,v,cv=3)
    accuracy = 'Tree accuracy ' + model_name + ', crss val max depth = ' + d + ': '  + str(crss_val.mean())
    print(maxdep)
    print(accuracy)
    print('-' * 40)
    return model

    
def compare_regressors(X,y,model_name):
    # this function takes a training data frame, a labels vector, and list containing a DecisionTreeRegressor and a RandomForestRegressor. 
    crss_val = cross_val_score(model_name[0], X, y, scoring = "neg_mean_squared_error", cv = 3)
    val = np.sqrt(-crss_val)
    avg_rmse = round(val.mean(), 1)
    std_dev =  round(val.std(ddof=1), 1)
    print("Model: DecisionTreeRegressor")
    print("Average of the RMSE's for the 3 folds: " + str(avg_rmse))
    print("Std dev of the RMSE's for the 3 folds: " + str(std_dev))
    print("-" * 40)

    crss_val = cross_val_score(model_name[1], X, y, scoring = "neg_mean_squared_error", cv = 3)
    val = np.sqrt(-crss_val)
    avg_rmse = round(val.mean(), 1)
    std_dev =  round(val.std(ddof=1), 1)
    print("Model: RandomForestRegressor")
    print("Average of the RMSE's for the 3 folds: " + str(avg_rmse))
    print("Std dev of the RMSE's for the 3 folds: " + str(std_dev))
    print("-" * 40)
def predict(feature_vector, model_list, class_list):
    # this function takes a 2D numpy array representing a sequence of feature vectors, list of three models
    preds = []

    for i in range(len(model_list)):
        model = model_list[i]
        #model = model.fit(feature_vector)
        pred = model.predict(feature_vector)
        preds.append(pred[0])

    print("Predicted class: " + class_list[int(preds[0])])
    print("Predicted temperature, tree regressor: " + str(round(preds[1])))
    print("Predicted temperature, random forest regressor: " + str(preds[2]))
def main():
    # get the frame, clean it, get your training data and labels
    
    
    df = get_frame()
    clean_frame(df)
    training_data, labels = get_X_and_y(df)
    X = pd.get_dummies(training_data, prefix = {'color:color', 'shade:shade', 'finish:finish'})
    bins = bin_y(labels)
    class_name = ['cold','cool','medium','warm','hot']
    print()
    decisiontree1 = make_and_test(X,labels,'DecisionTreeClassifier',None,class_name,'temp')
    time.sleep(1)
    decisiontree2 = make_and_test(X,labels,'DecisionTreeClassifier',2,class_name,'temp2')
    time.sleep(1)
    decisionregressor1 = make_and_test(X,labels,'DecisionTreeRegressor',None,class_name,'temp3')
    time.sleep(1)
    decisionregressor2 = make_and_test(X,labels,'DecisionTreeRegressor',2,class_name,'temp4')
    time.sleep(1)
    Randomregressor1 = make_and_test(X,labels,'RandomForestRegressor',None,class_name,'temp5')
    time.sleep(1)
    Randomregressor2 = make_and_test(X,labels,'RandomForestRegressor',2,class_name,'temp6')
    print()
    com_re_li = [decisionregressor2,Randomregressor2]
    compare = compare_regressors(X,labels,com_re_li)
    print()
    vector = np.array([[126.7,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0]])
    predic = predict(vector,[decisiontree2,decisionregressor2,Randomregressor2],class_name)
    

if __name__ == "__main__":
    main()
     
    
    
    
    
