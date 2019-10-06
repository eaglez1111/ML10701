import numpy as np
from statistics import variance
import matplotlib.pyplot as plt


''' Constants Declare '''

eps = 10**(-9)
width = 15
cont_ft = [0,2,4,10,11,12]
disc_ft = [1,3,5,6,7,8,9,13]
Income = {'>50K':0, '<=50K':1, '>50K.':0, '<=50K.':1}

Workclass = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
Education = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
Marital = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
Occupation = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
Relationship = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
Race = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
Sex = ['Female', 'Male']
Country = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
Country_print = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad\\&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
Label = [ Workclass, Education, Marital, Occupation, Relationship, Race, Sex, Country ]
Label_print = [ Workclass, Education, Marital, Occupation, Relationship, Race, Sex, Country_print ]

featureName = ['age', 'workclass', 'fnlwgt', 'education', 'educationNum', 'marital', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
className = '``$>$50K\'\'', '``$\\leq$50K\'\''


def readFile(file='adult.data',length=10**6):
    data = np.loadtxt(file,dtype=str,delimiter=', ',skiprows=(file=='adult.test'))
    length = min(length,len(data))
    data = np.array( data[:length] )
    i = 0
    while i < len(data):
        if file=='adult.test':
            data[i,14] = data[i,14][:-1]
        deleted = 0
        for j in range(width):
            if data[i,j]=='?':
                data = np.delete(data,i,0)
                deleted = 1
                break
        if not deleted:
            i+=1
    return data

def groupData(data):
    length = len(data)
    d = [[],[]]
    for i in range(length):
        d[Income[data[i,14]]].append(data[i])
    return [np.array(d[0]),np.array(d[1])]

def getNB(data):
    length = [len(data[0]),len(data[1])]
    Py = 1.0*np.array(length)/np.sum(length)
    GausDist = [[[0,0]]*6,[[0,0]]*6]
    DiscProb = [ [ [],[],[],[],[],[],[],[] ] , [ [],[],[],[],[],[],[],[] ] ]
    for j in range(2):
        for i in range(6):
            arr = data[j][:,cont_ft[i]].astype(int)
            GausDist[j][i] = [np.mean(arr),variance(arr)+eps]
        for i in range(8):
            arr = np.array( data[j][:,disc_ft[i]] )
            for k in range(len(Label[i])):
                n = len(np.where(arr==Label[i][k])[0])
                DiscProb[j][i].append(1.0*n/length[j])
    def logP(X):
        def g_pdf(m,v,x):
            return 1.0/(np.sqrt(2*np.pi*v)) * np.exp( -1.0*(x-m)**2/2/v )
        lp = [np.log(Py[0]),np.log(Py[1])]
        for j in range(2):
            for i in range(6):
                p = g_pdf( GausDist[j][i][0],GausDist[j][i][1],float(X[cont_ft[i]]) )
                lp[j] += np.log(p)
            for i in range(8):
                p = DiscProb[j][i][Label[i].index(X[disc_ft[i]])]
                lp[j] += np.log(p)
        return lp
    return (Py, GausDist, DiscProb),logP

def test(logP,data,pr=0):
    y_guess, y_truth = [],[]
    for i in range(len(data)):
        lp = logP(data[i])
        y_guess.append(np.argmax(lp))
        y_truth.append(Income[data[i][-1]])
        if pr:
            print 'Line '+str(i)+': logP('+className[0]+')=',('%.4f'%lp[0]),\
                    ' , logP('+className[1]+')=',('%.4f\\\\'%lp[1])
    missRate = 1.0*np.sum(np.abs(np.array(y_truth)-np.array(y_guess)))/len(y_truth)
    return 1.0-missRate

def printParam((Py,gd,dp)):
    print 'P('+className[0]+')= %.4f\\\\'%Py[0]
    print 'P('+className[1]+')= %.4f'%Py[1]
    st = [ ['\\item ']*14, ['\\item ']*14 ]
    for j in range(2):
        for i in range(6):
            st[j][cont_ft[i]] += featureName[cont_ft[i]]+': mean='+'%.4f, '%gd[j][i][0] \
                                        +'var='+'%.4f'%gd[j][i][1]
            pass
        for i in range(8):
            st[j][disc_ft[i]] += featureName[disc_ft[i]]+': '
            for k in range(len(dp[j][i])):
                st[j][disc_ft[i]] += Label_print[i][k] + '=%.4f, '%dp[j][i][k]
        print '.\\\\Class '+className[j]+': \\begin{itemize}'
        for i in range(14):
            print st[j][i]
        print '\\end{itemize}'


''' This is the function for Q4.1 '''
def q4_1():
    trnSet = readFile()
    Param,logP = getNB(groupData(trnSet))
    printParam(Param)
    tstSet = readFile('adult.test',12)
    print test(logP,tstSet,1)

''' This is the function for both Q4.2a and Q4.2b '''
def q4_2ab():
    trnSet = readFile()
    Param,logP = getNB(groupData(trnSet))
    tstSet = readFile('adult.test')
    print 'Accuracy on training data: %.4f'%test(logP,trnSet)
    print 'Accuracy on testing data: %.4f'%test(logP,tstSet)


''' This is the function for Q4.2c '''
def q4_2c():
    n_set = 2**np.array(range(5,13+1))
    acc = np.zeros([2,9],dtype='float32')
    tstSet = readFile('adult.test')
    for i in range(9):
        trnSet = readFile('adult.data',n_set[i])
        trn_tstSet = [trnSet,tstSet]
        Param,logP = getNB(groupData(trnSet))
        for t in range(2):
            acc[t,i]=test(logP,trn_tstSet[t])
    print 'Accuracy:\n',acc

    plt.plot(n_set,acc[0],'.-',label='Training')
    plt.plot(n_set,acc[1],'.-',label='Testing')
    plt.xlabel('n')
    plt.ylabel('Accuracy')
    plt.title('Training/Testing w.r.t. n')
    plt.legend(loc='bottom right')
    plt.show()

q4_1()
#q4_2ab()
#q4_2c()
