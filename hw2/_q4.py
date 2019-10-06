import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.stats
import sys

bigVar = 10**15
eps = 10**(-9)

''' The constructing function of this class computes the answers for Q4.1b '''
class Attribute:
    def __init__(self, continuous, data, label=[]):
        self.continuous = continuous
        st = ''
        #print data
        if continuous:
            self.mean, self.var = np.mean(data),np.var(data)+eps
            st = 'mean='+('%.4f'%self.mean)+', var='+('%.4f'%self.var)
        else:
            #data += 1
            self.prob = data[:-1]/np.sum(data[:-1])
            for i in range(len(self.prob)):
                st += label[i]+'='+('%.4f'%self.prob[i])+', '
        print st

    def p(self,x):
        if self.continuous:
            result = scipy.stats.norm(self.mean, self.var).pdf(float(x))
        else:
            result = self.prob[x]
        return result

rich,poor = '``$>$50K\'\'', '``$\\leq$50K\'\''

Workclass = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked','?']
Education = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool','?']
Marital = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse','?']
Occupation = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces','?']
Relationship = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried','?']
Race = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black','?']
Sex = ['Female', 'Male','?']
Country = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands','?']
Income = ['>50K', '<=50K','?']
Label = [ [], Workclass, [], Education, [], Marital, Occupation, Relationship, Race, Sex, [], [], [], Country, Income ]
attributeType = [1,0,1,0,1,0,0,0,0,0,1,1,1,0,1]

_n = 14

def train(_N = 100000): # if 1:
    age =           [[],[]]
    workclass =     np.zeros([2,8+1])
    fnlwgt =        [[],[]]
    education =     np.zeros([2,16+1])
    educationNum =  [[],[]]
    marital =       np.zeros([2,7+1])
    occupation =    np.zeros([2,14+1])
    relationship =  np.zeros([2,6+1])
    race =          np.zeros([2,5+1])
    sex =           np.zeros([2,2+1])
    gain =          [[],[]]
    loss =          [[],[]]
    hours =         [[],[]]
    country =       np.zeros([2,41+1])
    income =        np.zeros([2],dtype='int32')
    attri = [age, workclass, fnlwgt, education, educationNum, marital,occupation,relationship,race,sex,gain,loss,hours,country,income]
    attriName = ['age', 'workclass', 'fnlwgt', 'education', 'educationNum', 'marital', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

    with open('adult.data') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',',doublequote='False', quotechar='', quoting=csv.QUOTE_NONE)
        i,cnt=0,0
        for row in spamreader:
            if cnt == _N:
                break
            cnt+=1
            for e in row:
                if e == '?' or e == ' ?':
                    row=[]
                    break
            if len(row):
                c = Income.index(row[14][1:])
                income[c]+=1
                for j in range(_n):
                    if attributeType[j]:
                        attri[j][c].append(float(row[j]))
                    else:
                        attri[j][ c, Label[j].index(row[j][1:]) ] += 1
                i+=1

        Py = [1.0*income[0]/np.sum(income),1.0*income[1]/np.sum(income)]
        print 'P('+rich+') =','%.4f'%Py[0], ', P('+poor+') =','%.4f'%Py[1]

    NB = [[],[]]
    for i in range(2):
        print '\n\n.\\\\Class '+poor+':' if i else 'Class '+rich+':'
        #Country[-5] = 'Trinadad\&Tobago'
        ''' The constructing functions inherently are invoked here with data imported from the file '''
        for j in range(_n):
            print '\n'+attriName[j]
            if attributeType[j]:
                NB[i].append(Attribute(1,attri[j][i]))
            else:
                NB[i].append(Attribute(0,attri[j][i],Label[j]))
    return (NB,Py)

def test((NB,Py),file='adult.test',_Nt=100000,pr=0): # if 1: #
    y_truth, y_guess = [],[]
    if file=='adult.test':
        skipfirstline = 1
        Income_ = [Income[0]+'.',Income[1]+'.']
    else:
        skipfirstline = 0
        Income_ = Income
    with open(file) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',',doublequote='False', quotechar='', quoting=csv.QUOTE_NONE)
        cnt=0
        for row in spamreader:
            if skipfirstline:
                skipfirstline = 0
                continue
            if cnt >= _Nt:
                break
            for e in row:
                if e == '?' or e == ' ?':
                    row=[]
                    break
            if len(row):
                c = Income_.index(row[14][1:])
                y_truth.append(c)
                lp = [np.log(Py[0]),np.log(Py[1])]
                for j in range(_n):
                    x = row[j]
                    if attributeType[j]==0:
                        x = Label[j].index(x[1:])
                    for i in range(2):
                        lp[i] += np.log(NB[i][j].p(x))
                y_guess.append(0 if lp[0]>=lp[1] else 1)
                if pr:
                    print '\nLine '+str(cnt),':'
                    print 'logP('+rich+')=',('%.4f'%lp[0]),' , logP('+poor+')=',('%.4f'%lp[1])
                cnt+=1
    if pr:
        print 'Y compare:\n',y_truth,'\n',y_guess
    hitRate = 1.0-1.0*np.sum(np.abs(np.array(y_truth)-np.array(y_guess)))/len(y_truth)
    return hitRate

''' This is the function that computes answers for Q4.1 '''
def q4_1():
    print test(train(),10,1)

''' This is the function that computes answers for both Q4.2a and Q4.2b '''

def q4_2ab():
    NB = train()
    print 'Accuracy on training data:\n', test(NB,'adult.data')
    print 'Accuracy on testing data:\n', test(NB,'adult.test')

''' This is the function for Q4.2c '''
def q4_2c():
    n_set = 2**np.array(range(5,13+1))
    acc = np.zeros([2,9],dtype='float32')
    file = ['adult.data','adult.test']
    for i in range(len(n_set)):
        NB = train(n_set[i])
        for t in range(2):
            acc[t,i]=test(NB,file[t])
            print i,n_set[i],t,acc[t,i]
    print 'Accuracy:\n',acc

    plt.plot(n_set,acc[0],'.-',label='Training')
    plt.plot(n_set,acc[1],'.-',label='Testing')
    plt.xlabel('n')
    plt.ylabel('Accuracy')
    plt.title('Training/Testing w.r.t. n')
    plt.legend(loc='bottom right')
    plt.show()


q4_1()
q4_2ab()
#q4_2c()
