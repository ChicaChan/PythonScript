########################################################################
#  rm(list=ls())清空控制台所有变量                                     #
#  .libPaths()查看程序包路径                                           #
#  openxlsx需要下载Rtools,方法:                                        #
#  1、安装installr包【install.packages("installr")】                   #
#  2、执行命令installr::install.rtools()                               #
#  3、下载完成后安装,安装过程中除了环境变量那里打勾外其他全部默认即可  #
########################################################################
rm(list=ls())
#设置工作目录
setwd("D:\\DP小工具\\5.挑样本\\V4")
library(Rglpk)
library(openxlsx)
#读取数据以及条件
all<-read.xlsx("input.xlsx",sheet = 1)
raw_data<-as.data.frame(all[,2:ncol(all)])
quota_limit<-read.xlsx("input.xlsx",sheet = 2)
#计算列变量个数
rd_col<-ncol(raw_data)
#分组统计并生成变量x
if(rd_col>1){
  data<-aggregate(raw_data[,1],raw_data[,1:rd_col],length)
}else{
  data<-data.frame(table(raw_data[,1]))
  names(data)[1] <- names(all)[2]
  names(data)[2] <- "x"
}


#重命名计算结果的列名
names(data)[rd_col+1]<-"raw"
#生成增加列变量x，值为x1...xi
for(i in 1:length(data$raw)){
  data$x[i]<-paste("x",i,sep="")
}
#计算条件个数
n<-nrow(quota_limit)
#每个条件对应的xi变量
for(i in 1:n){
  assign(paste("con",i,sep=""),eval(parse(text=paste(quota_limit[i,1]))))
}
#做出m*n的矩阵（m代表所有条件的数量，n代表可变单元格数量）
aa<-(paste("x",seq(1:nrow(data)),sep=""))
bb<-matrix(c(rep(aa,n)),nrow=n,byrow=T)

#自建函数
rsl<-function(data){
  obj<-c(rep(1,nrow(data)))
  for(i in 1:n){
    bb[i,which(bb[i,] %in% eval(parse(text=paste("con",i,sep=""))))]<-1 #将coni中有的元素批量替换为1
    bb[i,which(regexpr("x",bb[i,])>0)]<-0 #将包含"x"的单元格批量替换为0
  }
  bb1<-bb
  bb2<-rbind(bb,bb1)
  cc<-diag(1,nrow=nrow(data))#可变单元格的系数矩阵
  mat<-rbind(bb2,cc)#完整的系数矩阵
  dir<-c(rep(c("<=",">="),each=n),rep("<=",nrow(data)))#符号
  rhs1<-append(quota_limit$max,quota_limit$min)
  rhs<-t(append(rhs1,data$raw))#转置所有限制配额(append:追加数据，这里不能用rbind)
  ls<-Rglpk_solve_LP(obj,mat,dir,rhs,max=T,types = "I")#求解
  #组合编码
  if(ncol(data)==3){
    data$combine<-data[,1]
  }else{
    data$combine<-do.call(paste,c(data[,1:(ncol(data)-2)],sep=""))
  }
  
  result<-cbind(data,ls$solution)
  names(result)[ncol(result)]<-"solution"
  return(result)
}
result<-rsl(data)#规划结果

#下面对规划结果挑选样本
if(ncol(data)==3){
  combine<-all[,2]
}else{
  combine<-do.call(paste,c(all[,2:ncol(all)],sep=""))
}

#自定义函数：条件计数，只统计到目前记录已有的，不是全部
cf<-function(array){
  accum<- c()
  n<-rep(1,length(array))#由于只能累加运算，故增加全部为1的n变量变相累计
  temp<-tapply(n,array,cumsum)#对array累加
  a<-sort(array[!duplicated(array)])#去重并顺序排列
  i<-length(a)
  for(i in 1:i){
    accum[which(array==a[i])] <- temp[[i]]#将累加值附入数组
  }
  accum<-cbind(array,accum)
  return(as.data.frame(accum))
}
accum<-cf(combine)
names(accum)[1]<-"combine"
all<-cbind(all,accum)
all<-merge(all,result[,(ncol(result)-1):(ncol(result))],by="combine")
#已有数小于规划数，标为1，否则为0
all$select[as.numeric(as.character(all$accum))<=all$solution]<-1#因子必须经过双重转化为数值
all$select[as.numeric(as.character(all$accum))>all$solution]<-0
#目标值和规划值对比
myfun<-function(df,default){
  if(default==1){
    nc<-ncol(result)
    df1<-gsub("\\$x","",df) #"//":转义符,否则$将被视为正则表达式
    df2<-gsub("data","result",df1)
    df3<-gsub("]",paste(",",nc,"]",sep=""),df2)
  }
  if(default==0){
    nc<-ncol(result)-3
    df1<-gsub("\\$x","",df) #"//":转义符,否则$将被视为正则表达式
    df2<-gsub("data","result",df1)
    df3<-gsub("]",paste(",",nc,"]",sep=""),df2)
  }
  return(df3)
}
compare<-quota_limit[,1:2]
compare$raw<-0
compare$raw_gap<-0
compare$plan<-0
compare$plan_gap<-0

for(i in 1:n){
  if(compare[i,1]!="data$x"){
  compare[i,3]<-sum(eval(parse(text=myfun(compare[i,1],0))))
  compare[i,4]<-compare[i,3]-compare[i,2]
  compare[i,5]<-sum(eval(parse(text=myfun(compare[i,1],1))))
  compare[i,6]<-compare[i,5]-compare[i,2]
  compare[i,1]<-gsub("data\\$x","",compare[i,1])
  }
  if(compare[i,1]=="data$x"){
    compare[i,3]<-sum(eval(parse(text=paste(myfun(compare[i,1],0),"[,",ncol(result)-3,"]",sep=""))))
    compare[i,4]<-compare[i,3]-compare[i,2]
    compare[i,5]<-sum(eval(parse(text=paste(myfun(compare[i,1],1),"[,",ncol(result),"]",sep=""))))
    compare[i,6]<-compare[i,5]-compare[i,2]
    compare[i,1]<-"result"
  }
}

#去除中间变量
samples_select<-subset(all,select=-c(combine,accum,solution))
samples_select[,1] <- paste(samples_select[,1],";",sep="")
result<-subset(result,select=-c(x,combine))
#结果输出
wb<-createWorkbook("Creator of workbook")#创建一个新工作簿
sheet_name<-c("samples_select","result","compare")

for (i in 1:3) {
  addWorksheet(wb, sheetName = sheet_name[i])#在创建的工作簿中增加sheet
  writeData(wb,sheet=i,eval(parse(text=sheet_name[i])))#分别写入数据到sheet中
  freezePane(wb, sheet_name[i] ,  firstActiveRow =2)#冻结首行
  if(i!=3){
    addStyle(wb,sheet=i, createStyle(fgFill="sienna2",halign="center"), rows=1, cols=1:ncol(eval(parse(text=sheet_name[i]))))
    addStyle(wb,sheet=i, createStyle(halign="center"), rows=2:(nrow(eval(parse(text=sheet_name[i])))+1), cols=1:6,gridExpand=T)
    
  }else{
    addStyle(wb,sheet=i, createStyle(fgFill="sienna2",halign="center"), rows=1, cols=1:6)
    addStyle(wb,sheet=i, createStyle(halign="center"), rows=2:(nrow(eval(parse(text=sheet_name[i])))+1), cols=1:6,gridExpand=T)
    setColWidths(wb, sheet=i, cols=1, widths = "auto")
  }
}

saveWorkbook(wb, file = "output.xlsx", overwrite = TRUE) #输出excel文件

if(sum(compare$plan)>0){
  print("Success!")
}else{
  print("Fail~")
}
