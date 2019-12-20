#This is the code for the SDSC5001 Statistical Machine Learning Final Project

#read data
data=read.table('desktop/pop_failures.dat', header=TRUE)
names(data)
 [1] "Study"                "Run"                  "vconst_corr"        
 [4] "vconst_2"             "vconst_3"             "vconst_4"           
 [7] "vconst_5"             "vconst_7"             "ah_corr"            
[10] "ah_bolus"             "slm_corr"             "efficiency_factor"  
[13] "tidal_mix_max"        "vertical_decay_scale" "convect_corr"       
[16] "bckgrnd_vdc1"         "bckgrnd_vdc_ban"      "bckgrnd_vdc_eq"     
[19] "bckgrnd_vdc_psim"     "Prandtl"              "outcome"  

#find number of failures, which is 46
dim(subset(data,outcome==0))
[1] 46 21

#exploratory analysis, for the first feature vconst_corr, for example
boxplot(data[,20]~data$outcome, col = 'red',varwidth = TRUE,
       xlab = 'outcome',ylab = colnames(data)[20],
       main = paste(colnames(data)[20],' vs outcome'))

#logistic regression feature set A and B
glm1.fit=glm(outcome~vconst_corr+vconst_2+vconst_4+vconst_5+vconst_7+slm_corr+convect_corr+bckgrnd_vdc1+bckgrnd_vdc_psim, data=data, family=binomial)
glm2.fit=glm(outcome~vconst_3+ah_corr+ah_bolus+efficiency_factor+tidal_mix_max+vertical_decay_scale+bckgrnd_vdc_ban+bckgrnd_vdc_eq+Prandtl, data=data, family=binomial)

# Split Data into Training and Testing
sample_size = floor(0.6*nrow(data))
set.seed(777)
index = sample(seq_len(nrow(data)),size = sample_size)
train =data[index,]
test =data[-index,]

# Testing logistic models
glm1.fit=glm(outcome~vconst_corr+vconst_2+vconst_4+vconst_5+vconst_7+slm_corr+convect_corr+bckgrnd_vdc1+bckgrnd_vdc_psim, data=train, family=binomial)
glm2.fit=glm(outcome~vconst_3+ah_corr+ah_bolus+efficiency_factor+tidal_mix_max+vertical_decay_scale+bckgrnd_vdc_ban+bckgrnd_vdc_eq+Prandtl, data=train, family=binomial)
glm1.probs=predict(glm1.fit,test,type='response')
glm2.probs=predict(glm2.fit,test,type='response')
glm1.pred=rep('crash',216)
glm2.pred=rep('crash',216)
glm1.pred[glm1.probs>.5]='nocrash'
glm2.pred[glm2.probs>.5]='nocrash'
table(glm1.pred,test[,21])
table(glm2.pred,test[,21])

#testing LDA models
library(MASS)
lda1.fit=lda(outcome~vconst_corr+vconst_2+vconst_4+vconst_5+vconst_7+slm_corr+convect_corr+bckgrnd_vdc1+bckgrnd_vdc_psim, data=train)
lda2.fit=lda(outcome~vconst_3+ah_corr+ah_bolus+efficiency_factor+tidal_mix_max+vertical_decay_scale+bckgrnd_vdc_ban+bckgrnd_vdc_eq+Prandtl, data=train)
lda1.pred=predict(lda1.fit,test)
lda2.pred=predict(lda2.fit,test)
lda1.class=lda1.pred$class
lda2.class=lda2.pred$class
table(lda1.class,test[,21])
table(lda2.class,test[,21])

# testing QDA models
qda1.fit=qda(outcome~vconst_corr+vconst_2+vconst_4+vconst_5+vconst_7+slm_corr+convect_corr+bckgrnd_vdc1+bckgrnd_vdc_psim, data=train)
qda2.fit=qda(outcome~vconst_3+ah_corr+ah_bolus+efficiency_factor+tidal_mix_max+vertical_decay_scale+bckgrnd_vdc_ban+bckgrnd_vdc_eq+Prandtl, data=train)
qda1.class=predict(qda1.fit,test)$class
qda2.class=predict(qda2.fit,test)$class
table(qda1.class,test[,21])
table(qda2.class,test[,21])

# testing KNN models
library(class)
train1=train[c('vconst_corr','vconst_2','vconst_4','vconst_5','vconst_7','slm_corr','convect_corr','bckgrnd_vdc1','bckgrnd_vdc_psim')]
train2=train[c('vconst_3','ah_corr','ah_bolus','efficiency_factor','tidal_mix_max','vertical_decay_scale','bckgrnd_vdc_ban','bckgrnd_vdc_eq','Prandtl')]
test1=test[c('vconst_corr','vconst_2','vconst_4','vconst_5','vconst_7','slm_corr','convect_corr','bckgrnd_vdc1','bckgrnd_vdc_psim')]
test2=test[c('vconst_3','ah_corr','ah_bolus','efficiency_factor','tidal_mix_max','vertical_decay_scale','bckgrnd_vdc_ban','bckgrnd_vdc_eq','Prandtl')]
train.outcome=train[,21]
set.seed(1)
knn1.pred=knn(train1,test1,train.outcome,k=2)
knn2.pred=knn(train2,test2,train.outcome,k=2)
table(knn1.pred,test[,21])
table(knn2.pred,test[,21])

#transforming features with LOOCV
acc=rep(0,nrow(data))
for(i in 1:nrow(data))
{
    p=1
    train=data[-i,]
    test=data[i,]
    glm.fit=glm(outcome~poly(vconst_corr,p)+vconst_2+vconst_4+vconst_5+vconst_7+slm_corr+poly(convect_corr,p)+bckgrnd_vdc1+bckgrnd_vdc_psim, data=train, family=binomial)
    
    results_prob=predict(glm.fit,test,type='response')
    results=ifelse(results_prob > 0.5,1,0)
    answers=test[,21]
    misClasificError=mean(answers != results)
    acc[i]=1-misClasificError
}
mean(acc)

# compare degree 1 and 2
glm1.fit=glm(outcome~vconst_corr+vconst_2+vconst_4+vconst_5+vconst_7+slm_corr+convect_corr+bckgrnd_vdc1+bckgrnd_vdc_psim, data=train, family=binomial)
glm2.fit=glm(outcome~poly(vconst_corr,2)+vconst_2+vconst_4+vconst_5+vconst_7+slm_corr+poly(convect_corr,2)+bckgrnd_vdc1+bckgrnd_vdc_psim, data=train, family=binomial)
glm1.probs=predict(glm1.fit,test,type='response')
glm2.probs=predict(glm2.fit,test,type='response')
glm1.pred=rep('crash',216)
glm2.pred=rep('crash',216)
glm1.pred[glm1.probs>.5]='nocrash'
glm2.pred[glm2.probs>.5]='nocrash'
table(glm1.pred,test[,21])
table(glm2.pred,test[,21])