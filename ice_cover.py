#Name Gus Kalivas

import csv
import math
import numpy as np
import random 

'''
takes no arguments and returns the data as described below in an n-by-2 array
'''
def get_dataset():
    full = []  # creates a list 
    # opens the csv file and reads in each line
    with open('full_freeze_ice_covers.csv', encoding = 'UTF-8') as f:
        reader = csv.DictReader(f)
        # skips over empty lines
        for row in reader:
            #the CSV file is giving me weird symbols infront of 'WINTER' it wouldnt let me remove
            if row['\ufeffWINTER'][:4] == "" or row['DAYS'] == "":
                continue
            #makes a list of lists
            year = []
            year.append(row['\ufeffWINTER'][:4])
            year.append(row['DAYS'])
            full.append(year)
            
    
    return full

'''
 takes the dataset as produced by the previous function and prints several statistics about the data
 does not return anything
'''
def print_stats(dataset):
    tot = 0 # add up all y values 
    tot2 = 0
    s = []
    # adds all the y values and casts to ints
    for x in dataset:
        tot+= int(x[1])
        s.append(int(x[1]))
    #calculates that average
    mean = tot/len(dataset)
    # calculates the std
    for y in s:
        tot2 += (y - mean)**2
    
    std = math.sqrt(tot2*(1/(len(dataset) -1)))
    #prints number of data points, mean and standard deviation
    print(len(dataset))
    print('%.2f' % mean)
    print('%.2f' % std)

'''
calculates and returns the mean squared error on the dataset given fixed betas
'''
def regression(beta_0, beta_1, dataset=get_dataset()):
    m = 0 # loops through the data set and calc mean squared error 
    for i in dataset:
        m+= (beta_0 + beta_1*float(i[0]) - float(i[1]))**2
    
    return m/len(dataset)

'''
performs a single step of gradient descent on the MSE and returns the derivative values as a tuple
'''
def gradient_descent(beta_0, beta_1, dataset=get_dataset()):
    x1 = 0 #sum for beta 0
    x2 = 0 #summ for beta 1
    #calc graidient descent for beta_0 and beta_1
    for i in dataset:
        x1+= (beta_0 + beta_1*(float(i[0])) - float(i[1]))
    for i in dataset:
        x2+= (beta_0 + beta_1*(float(i[0])) - float(i[1]))*float(i[0])
    
    g1 = x1*(2/len(dataset))
    g2 = x2*(2/len(dataset))
    #returns derivates vlaues in a tuple
    return (g1, g2)

'''
performs T iterations of gradient descent starting at 0,0 with the given parameter
and prints the results; does not return anything
'''
def iterate_gradient(T, eta):
    #get first two derivatives 
    g1, g2 = gradient_descent(0,0)
    b1 = 0- eta*g1 #caculates first beta 0
    b2 = 0 -eta*g2 #cacluates first beta 1
    #loop T times 
    for i in range(1, T+1):
        #calc regression for beta 0,t and beta 1,t
        r = regression(b1,b2)
        #prints results
        print(i, '%.2f' % b1,'%.2f' % b2, '%.2f' % r)
        #gets next derivatives and calc new beta 0, beta 1
        g1, g2 = gradient_descent(b1,b2)
        b1 = b1 - eta*g1
        b2 = b2- eta*g2
   
    return None

'''
using the closed-form solution, calculates and returns the values of B0,B1
and the corresponding MSE as a three-element tuple
'''
def compute_betas():
    x= get_dataset() # gets the dataset
    b1 = 0
    b0 = 0
    x1 = 0
    y1 = 0
    # calculates x and y average
    for i in x:
        x1 += int(i[0])
        y1 += int(i[1])
    x_bar = x1/len(x)
    y_bar = y1/len(x)
    s1 = 0
    s2 = 0
    #calculates beta using closed-form solution/equation
    for j in x:
        s1 += (int(j[0]) - x_bar)*(int(j[1]) - y_bar)
        s2 += (int(j[0]) - x_bar)**2

    b1 = s1/s2
    b0 = y_bar -b1*x_bar
    #returns regression from b0, b1
    r = regression(b0, b1)
    return (b0,b1, r) #returns b0,b1 and regression as a tuple
        
'''
using the closed-form solution betas, return the predicted number of ice days for that year
'''
def predict(year):
    #computes beta using closed form
    tup = compute_betas()
    return (tup[0] + tup[1]*year) #return prediction

'''
normalizes the data before performing gradient descent, prints results
'''
def iterate_normalized(T, eta):
    x = get_dataset() #gets original dataset
    x1 = 0
    s1 = 0
    #calculate the average and std
    x_val = []
    for i in x:
        x1 += int(i[0])
        x_val.append(int(i[0]))
    x_bar = x1/(len(x))


    for j in x_val:
        s1+= (float(j) - x_bar)**2

    std = s1*(1/(len(x) -1))
    std = math.sqrt(std)
    #makes new dataset with normalized x values and same y values
    dat =[]
    for y in x:
        dat.append([(int(y[0]) -x_bar)/std, int(y[1])])

    #get deriviates for 0,0 
    g1, g2 = gradient_descent(0, 0, dat) #with normalized data 'dat'
    #calculates new betas 
    b1 = 0- eta*g1
    b2 = 0 -eta*g2
    #for t iterations calculates gradiant descent and new betas with normalized dataset
    for i in range(1, T+1):
        r = regression(b1,b2, dat)
        print(i, '%.2f' % b1,'%.2f' % b2, '%.2f' % r)
        g1, g2 = gradient_descent(b1,b2, dat) #using normailized data 'dat'
        b1 = b1 - eta*g1
        b2 = b2 - eta*g2
    return None

'''
Performs stochastic gradient descent, prints results 
'''
def sgd(T, eta):
    x = get_dataset() #gets original data and normalizes it
    x1 = 0
    s1 = 0
    x_val = []
    #calculates average and std
    for i in x:
        x1 += int(i[0])
        x_val.append(int(i[0]))
    x_bar = x1/(len(x))

    for j in x_val:
        s1+= (float(j) - x_bar)**2
    std = math.sqrt(s1*(1/(len(x) -1)))
    dat =[] #create new dataset with normailized x values and same y values
    for y in x:
        dat.append([((float(y[0]) -x_bar)/std), int(y[1])])
    #gets  a random data point

    num = random.randint(0,len(dat)-1) #gets random data point to use
    #calculates deriviates with stochastic gradient descent 
    g1 = 2*(0+ 0*dat[num][0] - dat[num][1])
    g2 = 2*(0+ 0*dat[num][0] - dat[num][1])*dat[num][0]
    b1 = 0- eta*g1
    b2 = 0 -eta*g2
    #for t iterations calculates betas and regression with random point in dataset
    for i in range(1, T+1):
        r = regression(b1,b2, dat)
        print(i, '%.2f' % b1,'%.2f' % b2, '%.2f' % r) # print results 
        num = random.randint(0,len(dat) -1)
        g1 = 2*(b1+ b2*dat[num][0] - dat[num][1])
        g2 = 2*(b1+b2*dat[num][0] - dat[num][1])*dat[num][0]
        b1 = b1 - eta*g1
        b2 = b2- eta*g2
   

if __name__=="__main__":
    x = get_dataset()
    print_stats(x)
    x = regression(200,-.2)
    
    y =gradient_descent(200,-.2)
    z = iterate_gradient(5, 1e-7)
    c = compute_betas()
    print(c)
    p = predict(2021)
    print(p)
    
    n = iterate_normalized(5, 0.1)
    st = sgd(5, .1)

