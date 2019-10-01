
for i in range(10):
    print i
    if i>3:
        break

'''

import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.stats

class Attribute:
    def __init__(self, continuous, data, label=[]):
        self.continuous = continuous
        st = ''
        if continuous:
            self.mean, self.var = np.mean(data),np.var(data)
            st = 'mean='+str(self.mean)+', var='+str(self.var)
        else:
            data += 1
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

Workclass = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked','?']
Education = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool','?']
Marital = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse','?']
Occupation = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces','?']
Relationship = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried','?']
Race = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black','?']
Sex = ['Female', 'Male','?']
Country = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands','?']
Income = ['>50K', '<=50K','?']

_N = 32562
_n = 14

#def train():
if 1:
    age =           np.empty([2,_N],dtype='float32')
    workclass =     np.zeros([2,8+1])
    fnlwgt =        np.empty([2,_N],dtype='float32')
    education =     np.zeros([2,16+1])
    educationNum =  np.empty([2,_N],dtype='float32')
    marital =       np.zeros([2,7+1])
    occupation =    np.zeros([2,14+1])
    relationship =  np.zeros([2,6+1])
    race =          np.zeros([2,5+1])
    sex =           np.zeros([2,2+1])
    gain =          np.empty([2,_N],dtype='float32')
    loss =          np.empty([2,_N],dtype='float32')
    hours =         np.empty([2,_N],dtype='float32')
    country =       np.zeros([2,41+1])
    income =        np.zeros([2],dtype='int32')

    with open('adult.data') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        i=0
        for row in spamreader:
            if i == _N:
                break
            for e in row:
                if e == '?' or e == ' ?':
                    row=[]
                    break
            if len(row):
                c = Income.index(row[14][1:])
                income[c]+=1
                age[c,i] = row[0]
                workclass[c,Workclass.index(row[1][1:])] += 1
                fnlwgt[c,i] = row[2]
                education[c,Education.index(row[3][1:])] += 1
                educationNum[c,i] =  row[4]
                marital[c,Marital.index(row[5][1:])] += 1
                occupation[c,Occupation.index(row[6][1:])] += 1
                relationship[c,Relationship.index(row[7][1:])] += 1
                race[c,Race.index(row[8][1:])] += 1
                sex[c,Sex.index(row[9][1:])] += 1
                gain[c,i] = row[10]
                loss[c,i] = row[11]
                hours[c,i] = row[12]
                country[c,Country.index(row[13][1:])] += 1
                i+=1
        Py = income/np.sum(income)
        print 'P(''<=50K'') =',Py[0], ', P(''>50K'') =',Py[1]

    NB = [[],[]]
    for i in range(2):
        print '\n\n.\\\\Class ``$>$50K\'\':' if i else 'Class ``$\\leq$50K\'\':'
        print '\nage:'
        NB[i].append(Attribute(1,age[i,:income[i]]))
        print '\nworkclass:'
        NB[i].append(Attribute(0,workclass[i],Workclass))
        print '\nfnlwgt:'
        NB[i].append(Attribute(1,fnlwgt[i,:income[i]]))
        print '\neducation:'
        NB[i].append(Attribute(0,education[i],Education))
        print '\neducation-num:'
        NB[i].append(Attribute(1,educationNum[i,:income[i]]))
        print '\nmarital-status:'
        NB[i].append(Attribute(0,marital[i],Marital))
        print '\noccupation:'
        NB[i].append(Attribute(0,occupation[i],Occupation))
        print '\nrelationship:'
        NB[i].append(Attribute(0,relationship[i],Relationship))
        print '\nrace:'
        NB[i].append(Attribute(0,race[i],Race))
        print '\nsex:'
        NB[i].append(Attribute(0,sex[i],Sex))
        print '\ncapital-gain:'
        NB[i].append(Attribute(1,gain[i,:income[i]]))
        print '\ncapital-loss:'
        NB[i].append(Attribute(1,loss[i,:income[i]]))
        print '\nhours-per-week:'
        NB[i].append(Attribute(1,hours[i,:income[i]]))
        print '\nnative-country:'
        Country[-5] = 'Trinadad\&Tobago'
        NB[i].append(Attribute(0,country[i],Country))


if 1:#def test():
    y_truth, y_guess = [],[]
    _Nt = 10
    with open('adult.data') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        i=0
        for row in spamreader:
            if i == _Nt:
                break
            for e in row:
                if e == '?' or e == ' ?':
                    row=[]
                    break
            if len(row):
                c = Income.index(row[14][1:])
                y_truth.append(c)
                lp = [0,0]
                for j in range(_n):
                    if row[j][0]==' ':
                        row[j]=row[j][1:]
                    for i in range(2):
                        lp[i] += NB[i][j].p(row[j])
                y_guess.append(0 if lp[0]>=lp[1] else 1)
                i+=1

print y_truth
print y_guess



def main():
    return 1


if __name__ == '__main__':
    main()
'''
