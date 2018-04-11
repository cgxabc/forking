library("ctv") 
library("tseries")
library("quantmod")
library("zoo")
library(fBasics)
library(moments)
library(car)
library(astsa)
library(VGAM)
library(ghyp)

##1. (a)
# we download the Sprint Nextel Corp. index.
x= get.hist.quote(instrument = "S", start = "1990-01-02", end="2013-06-20",quote="AdjClose") 
head(x)
tail(x)  # by checking head and tail, we see the data starts from 1990-01-02 and ends at 2013-06-20.


plot(x, main="Sprint Nextel Corp.", col="blue", xlab="Time", ylab="S Price")
chartSeries(x,main="Sprint Nextel Corp.", col="blue", xlab="Time", ylab="S Price")

## (b) Show descriptive statistics and perform the test:  
#H_0: \mu=0, vs H_1: \mu not equal to 0.
x<-coredata(x)
length(x)  #5909
mean(x)    #12.8908
sd(x)  #  10.32845
skewness(x)  #1.818344 
basicStats(x)  # kurtosis = 3.709197
min(x)  #1.37
max(x)  #58.24
mykurtosis1 = function(x,excess=3) {  
	 m4 = mean((x-mean(x))^4) 
	 kurt = m4/(sd(x)^4)-excess  
	 kurt
}
mykurtosis1(x)  #3.709197 ( this is excess kurtosis, same as derived from command)

t.test(x)  #  p-value < 2.2e-16. We reject the null hypothesis that the mean is zero. So the conclusion is that the true mean is not zero at 5% significance level.

##(c) Plot a histogram with a density function. Perform KS test to find the distribution.
hist(x, border="pink", breaks=seq(0,65, by=0.7),xlab="S price", main="Histogram of Sprint Nextel Corp. price",freq=FALSE)
lines(density(x))
ks.test(x,"pnorm")  #p-value < 2.2e-16. So it's not a normal at 5% significance level, which is also obvious.
ks.test(x,"pt",4)  # not t-distribution either.
normalTest(x,method="jb")  #p Value: < 2.2e-16 , not normal.


##(d)  Test for stationarity on x.
adf.test(x)  # p-value =0.6956 with alternative hypothesis being stationary. So the conclusion is that we cannot reject the null hypothesis, i.e. x is not stationary.

#Or we can calculate first two moments to examine the criteria of being stationary.

x1<-x[1:1000]  # select 1st to 1000th data.
mean(x1) #\mu_1
var(x1) #\sigma^2_1


x2<-x[2001:3000] # select 2001st to 3000th data.
mean(x2)  #\mu_2
var(x2)     #\sigma^2_2

# Use t-test to test H_0: \mu_1=\mu_2 vs H_1: otherwise.
t.test(x1,x2) # don't assume equal variance, p-value < 2.2e-16. So we reject the null hypothesis that the mean are the same for two subgroups.

# Use F-test to test: H_0: \sigma^2_1=\sigma^2_2 vs H_1: otherwise. Degrees of freedom are 1000,1000 respectively.

pf(var(x1)/var(x2) ,1000,1000)  # p-value=0 <0.05. Hence we reject the null hypothesis that the variances are equal for the two subgroups.

#hence we conclude x is not stationary, since mean is not constant everywhere, nor is the variance.

##(e) Transform the data x to a stationary data y, and test the stationarity.

x= get.hist.quote(instrument = "S", start = "1990-01-02", end="2013-06-20",quote="AdjClose") 
y<-diff(x) # take the difference of the price at t and t-1, such the the transformed data is stationary.
head(y)
plot(y,main="Difference of S price",ylab=c("Difference of S price"),col="orange", xlab="Time")
adf.test(y, alternative ="stationary") # p-value =0.01<0.05. Hence we reject the null hypothesis that y is not stationary. The conclusion is y is stationary.
kpss.test(y)  #p-value = 0.1. Hence we cannot reject the null hypothesis that y is level or trend stationary. The conclusion is y is stationary.


## (f) Show descriptive statistics and perform the test:  

#H_0: \mu=0, vs H_1: \mu not equal to 0.y<-ts(y)
length(y)  #5908
mean(y)    # E(y)=0.0001844956
var(y)  # var(y)=0.1742454
sd(y)  #  0.4174271
skewness(y)  # s(y)=0.02346547
basicStats(y)  # excess kurtosis = 18.683004
mykurtosis1(y)  # excess kurtosis k(y) = 18.683
min(y)  # -4.34
max(y)   # 3.95
t.test(y)  # p-value = 0.9729. We cannot reject the null hypothesis that the mean is 0 at 5% significance level. We conclude y has mean 0.

# plot a histogram with density density function for transformed data y.
hist(y,border="blue",breaks=200, main="Histogram of Sprint Nextel Corp.(transformed data)", xlab="S price(transformed)", freq=FALSE)
lines(density(y),col="red")

y<-ts(y)  #make y a time series
head(y)

# Try to fit a nig distribution. Estimate parameters using mle.
nig.fit<-fit.NIGuv(y)
nig.fit
summary(nig.fit) 
# alpha.bar           mu                sigma         gamma 
#0.094682168  0.001906830  0.427528775  -0.001712437

hist(nig.fit,breaks=200,ghyp.col='blue')

## (g)
par(mfrow=c(2,1))
acf(y,20)  #AR order is not zero
pacf(y,20)  # MA order is not zero

normalTest(y,method='jb')  #p-value<< 2.2e-16, hence y is not normal. The normal assumption is not satisfied.
qqPlot(y, distribution="norm",ylab="Time Series") 

##(h)
help(Box.test)
Box.test(y,lag=12,type="Ljung")  # p-value < 2.2e-16. Yes, there is serial correlation since the null hypothesis is independence.



##(i)


fitar<-sarima(y,1,0,0)
fitar  # fitted model is w(t)=-0.032 w(t-1)+e(t), where w(t)=y(t)-0.0002
#Coefficients:
#       ar1   xmean
#    -0.032  0.0002
#s.e.   0.013  0.0053
#AIC = -0.7478062,  AICc=-0.747467
# H_0: ar1=0 vs H1: ar1 not equal to 0.
#under normal assumption, 95% CI for ar1 is -0.032-1.96*0.013=-0.05748 to -0.032+1.96*0.013=-0.00652. Hence we reject the null hypothesis. The coefficient ar1 is significant.
resiar<-resid(fitar$fit)  #building the data set composed by the residuals of the fitting.
normalTest(resiar,method='jb')  #not normal
mean(resiar)  #-6.860916e-09
t.test(resiar)  #p-value = 1, cannot reject the null hypothesis that the mean is zero.
sd(resiar)  # 0.4172136< 0.4174271=sd(y). The model makes sense since the variance of residual is less than variance of Y.





fitma<-sarima(y,0,0,1)
fitma  # fitted model is y(t)=0.0002+e(t)-0.0308e(t-1).
#Coefficients:
#         ma1   xmean
#      -0.0308  0.0002
# s.e.   0.0128  0.0053
# AIC = -0.7477688, AICc=-0.7474296
# H_0: ma1=0 vs H1: ma1 not equal to 0.
#under normal assumption, 95% CI for ma1 is -0.0308-1.96*0.0128=-0.055888 to -0.0308+1.96*0.0128= -0.005712. Hence we reject the null hypothesis. The coefficient ma1 is significant.
resima<-resid(fitma$fit)  #building the data set composed by the residuals of the fitting.
head(resima)
normalTest(resima,method='jb')  #not normal
mean(resima)  #  -2.998694e-07
t.test(resima)  #p-value = 1, cannot reject the null hypothesis that the mean is zero.
sd(resima)  #  0.4172214 < 0.4174271=sd(y). The model makes sense since the variance of residual is less than variance of Y.



fitarma<-sarima(y,1,0,1)
fitarma  # fitted model is w(t)= -0.8558 w(t-1) + 0.8307 e(t-1) +e(t), where w(t)=y(t)-0.0002
#Coefficients:
#          ar1     ma1   xmean
#      -0.8558  0.8307  0.0002
# s.e.   0.0426  0.0454  0.0054
#  AIC = -0.7488199, AICc=-0.7484802
# H_0: ar1=0 vs H1: ar1 not equal to 0.
#under normal assumption, 95% CI for ar1 is -0.8558-1.96*0.0426= -0.939296 to -0.8558+1.96*0.0426=-0.772304. Hence we reject the null hypothesis. The coefficient ar1 is significant.
# H_0: ma1=0 vs H1: ma1 not equal to 0.
#under normal assumption, 95% CI for ma1 is -0.8307-1.96*0.0454=-0.919684 to -0.8307+1.96*0.0454= -0.741716. Hence we reject the null hypothesis. The coefficient ma1 is significant.

fitarma$fit  #sigma^2 estimated as 0.1738


fit1<-arima(y,order=c(1,0,1))  #same procedure as above
fit1
head(fit1$resid)  #same with resiarma
fit1$resid[5889]

fityes=y-fit1$resid
head(fityes)
head(y)


resiarma<-resid(fitarma$fit) #building the data set composed by the residuals of the fitting.
head(resiarma)
mean(resiarma)  #  -3.954052e-05
t.test(resiarma)  # p-value = 0.9942, cannot reject the null hypothesis that the mean is zero.
sd(resiarma)  #  0.4169316 < 0.4174271=sd(y). The model makes sense since the variance of residual is less than variance of Y.
var(resiarma)   #0.173832

par(mfrow=c(2,1))
acf(resiarma,200,main="ACF residuals of ARMA model")   # not all zeros are inside, but stay inside when n becomes large.
pacf(resiarma,200, main="PACF residuals of ARMA model")  # not exactly white noise
hist(resiarma,breaks=200,freq=FALSE, main="Histogram of residuals of ARMA(1,1) model")
normalTest(resiarma,method='jb')  #not normal


##(j)
## ARMA(1,1) has smallest AIC and AICc and the residual has the smallest variance, hence this model is the most fitted of the three.

##(k) Now we just focus on the ARMA(1,1) model.

resq<-resiarma^2  # residual squared
head(resq)
normalTest(resq,method='jb')  # p-value<2.2e-16   not normal
hist(resq,breaks=200)
var(resq)  #0.6149395


acf(resq,100)  #significant lags
pacf(resq,100)

fitresq<-sarima(resq,1,0,0)  # fit e(t)^2 with a AR(1) process
fitresq # fitted model is w(t)=0.3237 w(t-1)+u(t), where w(t)=e(t)^2-0.1738

#Coefficients:
#       ar1   xmean
#       0.3237  0.1738
#s.e.  0.0123  0.0143
#
# H_0: ar1=0 vs H1: ar1 not equal to 0.
#under normal assumption, 95% CI for ar1 is 0.3237-1.96*0.0123=0.299592  to 0.3237+1.96*0.0123=0.347808. Hence we reject the null hypothesis. The coefficient ar1 is significant.

#Hence there is an ARCH effect since ar1 coefficient is nonzero.

resu<-resid(fitresq$fit)
mean(resu)  #-8.272445e-06
t.test(resu)   # p-value = 0.9993 cannot reject the null hypothesis that mean is 0.
var(resu)  #0.5505026<0.6149395=var(resq)  So the model makes sense.
normalTest(resu,method='jb')  #not normal
acf(resu)
pacf(resu)  #not white noise


##(m)

## The model is  h(t)-0.0002= -0.8558 (h(t-1)-0.0002) + 0.8307 e(t-1) +e(t), 
# i.e. p(t+1)-p(t)=-0.8558( p(t)-p(t-1)-0.0002)+0.8307 e(t-1) + e(t),  p(t) is the predicted price
# i.e  p(t+1)=0.1442*p(t)+ 0.8558*p(t-1)+0.8307*e(t-1) +e(t)+0.000171, but e(t) is the error term from the difference between the price differece y and fitted value, i.e. y(t)-h(t) (replace h(t-1) with y(t-1) in the formula)


tail(y,20)
length(y)
ytail20=y[5889:5908]
ytail20  # y[i]=x[i+1]-x[i], true price difference vector, taking last 20 values

h<-numeric(20)
h[1]=0.01  #  set up a new price difference vector for predicting

e<-numeric(20)
e[1]=fit1$resid[5889]
e[1] #0.006841529  # set up an error term vector

for(i in 1:19)
{h[i+1]=0.0002-0.8558*(ytail20[i]-0.0002)+0.8307*e[i]
	e[i+1]=ytail20[i+1]-h[i+1]}
h

x<-coredata(x)
tail(x,20)
xtail21=x[5889:5909]  # true price vector, taking last 21 values
xtail21

p<-numeric(21)  # price vector for predicting the last 21 values
p[1]=7.30  # let the first value equal the true value

for(i in 1:20)
{p[i+1]=h[i]+p[i]}
p  # the result looks nice.






##2. (a)

# simulate AR(1) process with phi=0.5.
x1 = arima.sim(list(order=c(1,0,0), ar=0.5), n=5000) 
par(mfrow=c(3,1))
plot(x1,main=(expression(AR(1)~~~phi[1]==0.5)))
acf(x1,20)
pacf(x1,20)

# simulate AR(1) process with phi=1.2
e<-rnorm(5000,0,0.01)
x2<-numeric(5000)
x2[1]=0.1
for(i in 1:4999)
{x2[i+1]=1.2*x2[i]+e[i+1]}
x2
x2par<-x2[1:1000]  #choose the first 1000 data, otherwise it's not possible to plot acf and pacf in R. Note x2 is diverging.
par(mfrow=c(3,1))
plot(x2par, main=(expression(AR(1)~~~phi[1]==1.2)))
acf(x2par,20)
pacf(x2par,20) 

# (b)

# simulate MA(1) process with theta=0.7.
y1 = arima.sim(list(order=c(0,0,1), ma=0.7), n=5000) 
par(mfrow=c(3,1))
plot(y1,main=(expression(MA(1)~~~theta[1]==0.7)))
acf(y1,20)
pacf(y1,20)

# simulate MA(1) process with theta=3
e<-rnorm(5000,0,0.01)
y2<-numeric(5000)
y2[1]=e[1]
for( i in 1:4999)
{ y2[i+1]=e[i+1]+3*e[i]}
par(mfrow=c(3,1))
plot(y2,main=(expression(MA(1)~~~theta[1]==3)),type='l')
acf(y2,20)
pacf(y2,20)

# another way of simulating MA(1) with theta=3
y3 = arima.sim(list(order=c(0,0,1), ma=3), n=5000) 
par(mfrow=c(3,1))
plot(y3,main=(expression(MA(1)~~~theta[1]==3)))
acf(y3,20)
pacf(y3,20)

#(c)

# generating a mixture:

par(mfrow=c(2,2))
#w=0.25
Mix1 <- ifelse(runif(5000) < 0.25, rnorm(5000), rnorm(5000, 4, 0.6)) # ifelse(test, yes, no)
hist(Mix1,freq=F,breaks=70,main="0.25 N(0,1) +0.75 N(4,0.6)")
lines(density(Mix1),col="red")

#w=0.5
Mix2 <- ifelse(runif(5000) < 0.5, rnorm(5000), rnorm(5000, 4, 0.6)) # ifelse(test, yes, no)
hist(Mix2,freq=F,breaks=70,main="0.5 N(0,1) + 0.5 N(4,0.6)")
lines(density(Mix2),col="red")


#w=0.75
Mix3 <- ifelse(runif(5000) < 0.75, rnorm(5000), rnorm(5000, 4, 0.6)) # ifelse(test, yes, no)
hist(Mix3,freq=F,breaks=70, main="0.75 N(0,1) + 0.25 N(4,0.6)")
lines(density(Mix3),col="red")

#w=1
Mix4 <- ifelse(runif(5000) < 1, rnorm(5000), rnorm(5000, 4, 0.6)) # ifelse(test, yes, no)
hist(Mix4,freq=F,breaks=70, main ="N(0,1)")
lines(density(Mix4),col="red")


#(d)&(e)

x<-seq(-5,5,len=1000)
plot(x,dnig(x),type="l",ylim=c(0.0,1),col="red",xlab="x",ylab="density function f(x)")
lines(x,dnorm(x))
lines(x,dt(x,4),col="purple")

y = seq(-5, 5, by=0.01)
loc = 0; b = 1
lines(y, dlaplace(y, loc, b), type="l", col="green", ylim=c(0,1))

pal<-c("red","black","purple","green")
legend("topleft",c("NIG: alpha=1,beta=0,delta=1,mu=0","normal: mean=0, sd=1","t-dist:df=4","laplace:loc=0,scale=1"),lty=1,col=pal,bty="n",merge=TRUE, inset=0.1)


mykurtosis1(dnig(x))  #1.271777
mykurtosis1(dnorm(x))  #-0.3127756
mykurtosis1(dt(x,4))  #  -0.1414192
mykurtosis1(dlaplace(y,0,1))  #1.370897
# higher kurtosis, fatter tail, more peaked.

##(f)
e<-rnorm(5000,0,0.01)
ran<-numeric(5000)
ran[1]=e[1]
for(i in 1:4999)
{ran[i+1]=ran[i]+e[i+1]+1}
ran   # a random walk has the same correlation with a random walk without a drift.
par(mfrow=c(3,1))
plot(ran, main="random walk with drift 1")
acf(ran) #large nonvanishing spikes in acf
pacf(ran)




