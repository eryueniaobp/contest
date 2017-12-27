from __future__ import print_function
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import datetime,time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

LAG = 1
INPUT_SIZE = 1

"""
Solution 1:Use weather and weekday to assist. High performance.
           Add holiday ,monthday ,month-tag.
Solution 2:

"""

"""
overfitting > how to prevent overfitting  in  ppytorch .
"""
"""
WK max 132
9 +  6 :weekday + temp + weather
"""
WK = 9 + 12
row ,col = 15 , 40
class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.input_size = INPUT_SIZE
        self.hidden_size = 51
        self.lstm1 = nn.LSTMCell(1, self.hidden_size)
        self.atten = nn.Linear(self.hidden_size, 1)
        self.lstm3 = nn.Sigmoid()

        self.linear = nn.Linear(WK  ,1)

        # self.lr = nn.Sigmoid()

        wdf = pd.read_csv( '~/data/weather.fea.csv')
        wdf['date'] = pd.to_datetime(wdf['date'])

        # wdf = wdf [ wdf['date'] > '2016-08-31']

        d = datetime.datetime.strptime('2016-09-01','%Y-%m-%d')
        self.future_map = {}
        for  i in range(30):
            c = d + datetime.timedelta(days = i )


            self.future_map[i]  = autograd.Variable(  torch.from_numpy(
                wdf [ wdf['date'] == c ].values[:,1:1+WK].astype('float64') )
            )


    def forward(self, input, weather_input , future = 0):
        outputs = []
        h_t = Variable(torch.zeros(input.size(0), self.hidden_size).double(), requires_grad=False)
        c_t = Variable(torch.zeros(input.size(0), self.hidden_size).double(), requires_grad=False)

        # h_t2 = Variable(torch.zeros(input.size(0), 60).double(), requires_grad=False)
        # c_t2 = Variable(torch.zeros( input.size(0), 60).double(), requires_grad=False)

        h_t3 = Variable(torch.zeros(input.size(0), 1).double(), requires_grad=False)
        c_t3 = Variable(torch.zeros(input.size(0), 1).double(), requires_grad=False)

        # print( input.size(0) , input.size(1) )

        wmap  =  {}
        for i, weather_t in enumerate(weather_input.chunk( weather_input.size(1)   ,dim=1 ) ) :
            wmap[i] = weather_t
            # print(wmap[i].size())

        for i, input_t in enumerate(input.chunk(input.size(1)/self.input_size, dim=1)):
            # print ( i , input_t.data.size)
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            attn =  self.atten(c_t)
            # print (c_t.data.size() , h_t2.data.size() , c_t2.data.size() )
            # h_t2, c_t2 = self.lstm2(c_t, (h_t2, c_t2))
            # h_t3, c_t3 = self.lstm3(c_t, (h_t3, c_t3))
            c_t3 = self.lstm3(attn)


            # print ( input.size(0) , wmap[i].size(),wmap[i]  ,wmap[i].data.size() ,wmap[i].data   )
            # wea = wmap[i].data.numpy().reshape(input.size(0),WK)
            # wea = wmap[i].data.contiguous().view(input.size(0),WK)

            # print ( wea ,wmap[i] )

            # raise 'error'

            wea = wmap[i].squeeze(1)
            # print ( wea.size() )
            # print ( 'weasize = ' , wea.size() , ' c_t3 size = ' , c_t3.size() )
            # print (type(wea) , type(c_t3))
            # print ( torch.cat( (wea , c_t3.data ) ,dim=1 ) )

            # print (wea)
            # raise  'error'
            p =  self.linear(wea )
            # print ( 'train_p', p )
            # print (type(p))
            # o = torch.log( torch.nn.functional.relu( p * c_t3 ) )
            o = torch.nn.functional.relu( p * c_t3 )


            # print ( type(o))
            outputs += [o]

        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(outputs[-LAG ], (h_t, c_t))
            # h_t, c_t = self.lstm1(c_t3, (h_t, c_t))
            # h_t2, c_t2 = self.lstm2(c_t, (h_t2, c_t2))
            attn = self.atten(c_t)
            c_t3 = self.lstm3(attn)

            # print (  self.future_map[i].size() )

            # print(self.future_map[i])
            # wea = self.future_map[i].data.resize_(1,WK)
            wea = self.future_map[i]

            # print(wea)

            # raise 'error'
            # use future weather  to assist
            p = self.linear (wea )

            print ('future' , p.data )
            outputs += [ torch.nn.functional.relu (p*c_t3) ]
        outputs = torch.stack(outputs, 1).squeeze(2).float()
        return outputs


if __name__ == '__main__':
    # set ramdom seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    # data = torch.load('traindata.pt')
    df = pd.read_csv('~/data/merge.csv')



    input = Variable(torch.from_numpy(df['power_consumption'][9:].values.reshape(row,col)[:, :-LAG]) , requires_grad=False)
    weather_input = Variable(torch.from_numpy(df[range(3,3+WK)][9:].values.astype('float64').reshape(row,col, WK ) [:,LAG:,:]).contiguous() , requires_grad=False                             )
    print (weather_input.data.size())
    target = Variable(torch.from_numpy(df['power_consumption'][9:].values.reshape(row,col)[:, LAG:]), requires_grad=False)


    row_n = input.size(0)
    print ( row_n)
    # raise  'Error'
    # build the model
    seq = Sequence()
    seq.double()
    criterion = nn.MSELoss()
    # criterion = nn.KLDivLoss()
    # criterion = nn.L1Loss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(),tolerance_grad=1e-9,tolerance_change=1e-9)
    #begin to train
    for i in range(6):
        print('STEP: ', i)

        def closure():

            optimizer.zero_grad()
            out = seq(input,weather_input)


            out =  torch.log(out +1 )
            # print ( out )
            # raise 'error'

            # out = Variable( torch.Tensor.log(out.data.float() + 1 ) ,requires_grad=True)
            # print (out[0] )
            # print(out , target)
            loss = criterion(out, torch.log( target +1 ).float() )

            print(   'loss:', loss.data.numpy()[0])
            loss.backward()
            return loss
        optimizer.step(closure)
        # begin to predict
        future = 30/INPUT_SIZE

        pred = seq(input[  row_n-1 : row_n], weather_input[row_n -1 :row_n] , future = future)
        y = pred.data.numpy()

        # print (y)

        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        def draw(yi, color):
            print ( yi.shape)
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)/INPUT_SIZE].flatten(), color, linewidth = 2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future*INPUT_SIZE), yi[ (input.size(1)/INPUT_SIZE):].flatten(), color + '^', linewidth = 2.0)
        draw(y[0], 'r')

        # print ('-' * 10)
        # print (y[0][input.size(1):])



        # draw(y[1], 'g')
        # # draw(y[2], 'b')
        # print (input.size(1))
        # print (len(target))
        # print( len(target[11:12][0]))

        # # print (target.data[:3][0])
        # draw(target.data[:3][0].numpy(), 'b')
        #

        from scipy.ndimage.interpolation import shift


        plt.plot(np.arange(input.size(1)),
                 shift(target.data[row_n-1:row_n][0].numpy()[:input.size(1)],  0) ,
                 'b',
                 linewidth = 2.0)

        plt.plot(np.arange(input.size(1)),
                 shift(target.data[row_n-1:row_n][0].numpy()[:input.size(1)],  1) ,
                 'y',
                 linewidth = 2.0)
        plt.savefig('predict%d.pdf'%i)
        plt.close()

    def build_date(day,cnt):
        ds = []
        d = datetime.datetime.strptime(day,'%Y%m%d')
        for i in range(cnt):
            c = d + datetime.timedelta(days=i)
            ds.append(  c.strftime('%Y%m%d'))
        return ds
    p =  y[0][input.size(1)/INPUT_SIZE:].flatten() * 4905574.
    p =  [ int(i) for i in p ]
    pd.DataFrame({'predict_date': build_date('20160901',30) , 'predict_power_consumption': p }).to_csv('Tianchi_power_predict_table.csv',index=False)
