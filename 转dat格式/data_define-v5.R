rm(list=ls())
#options(digits = 22)
setwd("D:\\DP小工具\\2.转dat格式")
filename<-c("data")				#文件名字
raw<-read.csv(paste("./",filename,".csv",sep=""),stringsAsFactors = F)
raw[,which(lapply(raw,class)=='numeric')] <- round(raw[,which(lapply(raw,class)=='numeric')],5)
names(raw) <- gsub("[.]","0", names(raw))#将"."全部替换成0
attribute<-data.frame(colname=colnames(raw),type=0)
attribute[,1]<-gsub("_",0,attribute[,1])

for(i in 1:ncol(raw)){
  if(class(raw[,i])=="character"){
    attribute[i,2]<-"character"
  }
  if(class(raw[,i])=="numeric"){
    attribute[i,2]<-"numeric"
  }
  if(class(raw[,i]) %in% c("character","numeric")==F){
    attribute[i,2]<-"integer"
  }
}
raw[is.na(raw)]<-" "
##转dat格式
len <- c()
for(i in 1:ncol(raw)){
  len[i] <- max(nchar(raw[,i],type = "bytes"))
}
len[len<=6]<-6
attribute<-cbind(attribute,len,row.names =1:length(len))
#不直接用上面计算的,重新计算一次,因为两个for循环太慢,自定义一个函数快很多
fix_data<-function(df){
  df1<-c()
  len1<-max(nchar(df,type = "bytes"))
  len1[len1<=6]<-6
  for(i in 1:length(df)){
    df1[i]<-paste(paste(rep(" ",len1-nchar(df[i],type = "bytes")),collapse=""),df[i],sep="")
  }
  return(df1)
}
#用空格填充实现等宽数据
for(i in 1:ncol(raw)){
  raw[,i]<-fix_data(raw[,i])
}

##define.stp
s<-c()
for(i in 1:nrow(attribute)){
  s[i]<-(cumsum(attribute[1:i,3])+i)[i-1]
}
s[1]<-1
e<-s+attribute[,3]-1
attribute$define<-0
for(i in 1:nrow(attribute)){
  if(attribute[i,2]=="integer"){
    attribute[i,4]<-paste("di $",attribute[i,1],"=$",s[i],"-",e[i],",",sep="")
  }
  if(attribute[i,2]=="numeric"){
    attribute[i,4]<-paste("dw $",attribute[i,1],"=$",s[i],"-",e[i],",",sep="")
  }
  if(attribute[i,2]=="character"){
    attribute[i,4]<-paste("dc $",attribute[i,1],"=$",s[i],"-",e[i],",",sep="")
  }
}
##make.stp
colname1<-colnames(raw)
colname2<-gsub("_.*","",colname1)
colname2<-factor(colname2,unique(colname2))
colname_num<-as.data.frame(table(colname2))
m_colname<-colname_num[colname_num$Freq>1,]
make<-data.frame(df=paste(m_colname[,1],";",m_colname[,2],";",sep=""),stringsAsFactors = F)
if(make[1,1]==";;"){
  make<-"No multiple choice,please check your input data."
}else{
  make<-rbind(c("[*data ttl(;)="),make,"]","[*do i=1:[ttl.#]/2]","   [*do a=1:[ttl.i*2]]","      om $[ttl.i*2-1]=$[ttl.i*2-1]0[a]/1-999,","   [*end a]","[*end i]",sep="")
}
##输出结果
write.table(raw,paste(filename,".dat",sep=""),row.names = F,col.names = F,quote = F)
write.table(attribute[,4],paste(filename,"define.stp",sep=""),row.names = F,col.names = F,quote = F)
write.table(make,paste(filename,"make.stp",sep=""),row.names = F,col.names = F,quote = F)
attribute[attribute$type=="character",]#查看文本型字段


