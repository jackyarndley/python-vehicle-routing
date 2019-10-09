#Setting current working directory to source file location
rstudioapi::getActiveDocumentContext
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#Loading required libraries
library(tidyverse)
library(ggplot2)
library(s20x)

#Reading in data and cleaning
demand = read_csv("./data/demandData.csv", col_names=TRUE)
demand = gather(demand,key=date,value=demand,ends_with("19"))
demand$weekday = as.factor(weekdays(as.Date(demand$date,format="%d/%m/%Y")))
demand$Type <- sub("(\\w+).*", "\\1", demand$Supermarket)
demand$Type = as.factor(demand$Type)
demand$Supermarket = as.factor(demand$Supermarket)

#Filtering dataset by weekday/Saturday and brands
weekday <- filter(demand,weekday%in% c("Monday","Tuesday","Wednesday","Thursday","Friday"))
weekend <- filter(demand,weekday=="Saturday")
four.day <- filter(weekday,Type=="Four")
fresh.day <- filter(weekday,Type=="Fresh")
new.day <- filter(weekday,Type=="New")
pak.day <- filter(weekday,Type=="Pak")
four.end <- filter(weekend,Type=="Four")
fresh.end <- filter(weekend,Type=="Fresh")
new.end <- filter(weekend,Type=="New")
pak.end <- filter(weekend,Type=="Pak")

#Applying a linear model to each brand's Weekday and Saturday distribution for ANOVA testing
f.d.fit = lm(demand~Supermarket,data=four.day)
n.d.fit = lm(demand~Supermarket,data=new.day)
p.d.fit = lm(demand~Supermarket,data=pak.day)
f.e.fit = lm(demand~Supermarket,data=four.end)
n.e.fit = lm(demand~Supermarket,data=new.end)
p.e.fit = lm(demand~Supermarket,data=pak.end)

#Running ANOVA on data and collecting in a dataframe
a.d <- rbind(anova(f.d.fit)[1,],anova(f.e.fit)[1,],anova(n.d.fit)[1,],anova(n.e.fit)[1,],anova(p.d.fit)[1,],anova(p.e.fit)[1,])
anova.d <- data.frame(Brand = c("Four","Four","New","New","Pak","Pak"), Day = c("M-F","Sat","M-F","Sat","M-F","Sat"),P.value = a.d[,5])

#Setting demands and gathering summation data for four square
four.day.g <- group_by(four.day,Supermarket)
four.day.d <- data.frame(Supermarket=four.day$Supermarket[1:11],demand=ceiling(mean(four.day$demand)))
four.day.s <- summarise(four.day.g,sum=sum(demand))
four.end.g <- group_by(four.end,Supermarket)
four.end.d <- data.frame(Supermarket=four.end$Supermarket[1:11],demand=ceiling(mean(four.end$demand)))
four.end.s <- summarise(four.end.g,sum=sum(demand))

#Setting demands and gathering summation data for fresh
fresh.day.g <- group_by(fresh.day,Supermarket)
fresh.day.d <- data.frame(Supermarket=fresh.day$Supermarket[1],demand=ceiling(mean(fresh.day$demand)))
fresh.day.s <- summarise(fresh.day.g,sum=sum(demand))
fresh.end.g <- group_by(fresh.end,Supermarket)
fresh.end.d <- data.frame(Supermarket=fresh.end$Supermarket[1],demand=ceiling(mean(fresh.end$demand)))
fresh.end.s <- summarise(fresh.end.g,sum=sum(demand))

#Setting demands and gathering summation data for new world
new.day.g <- group_by(new.day,Supermarket)
new.day.d <- data.frame(Supermarket=new.day$Supermarket[1:19],demand=ceiling(mean(new.day$demand)))
new.day.s <- summarise(new.day.g,sum=sum(demand))
new.end.g <- group_by(new.end,Supermarket)
new.end.d <- data.frame(Supermarket=new.end$Supermarket[1:19],demand=ceiling(mean(new.end$demand)))
new.end.s <- summarise(new.end.g,sum=sum(demand))

#Setting demands and gathering summation data for pak'nsave
pak.day.g <- group_by(pak.day,Supermarket)
pak.day.d <- data.frame(Supermarket=pak.day$Supermarket[1:15],demand=ceiling(mean(pak.day$demand)))
pak.day.s <- summarise(pak.day.g,sum=sum(demand))
pak.end.g <- group_by(pak.end,Supermarket)
pak.end.d <- data.frame(Supermarket=pak.end$Supermarket[1:15],demand=ceiling(mean(pak.end$demand)))
pak.end.s <- summarise(pak.end.g,sum=sum(demand))


#Creating weekday planned demand and corresponding error dataframes
plan.day <- rbind(four.day.d,fresh.day.d,new.day.d,pak.day.d)
error.day <- rbind(four.day.s,fresh.day.s,new.day.s,pak.day.s)
error.day$demand <- plan.day$demand
error.day$error <- error.day$demand*20/error.day$sum
error.day$fraction <- abs(error.day$error-1)
error.day$sum <- sum(error.day$fraction)

#Creating Saturday planned demand and corresponding error dataframes
plan.end <- rbind(four.end.d,fresh.end.d,new.end.d,pak.end.d)
error.end <- rbind(four.end.s,fresh.end.s,new.end.s,pak.end.s)
error.end$demand <- plan.end$demand
error.end$error <- error.end$demand*4/error.end$sum
error.end$fraction <- abs(error.end$error-1)
error.end$sum <- sum(error.end$fraction)

#Saving data as csv for further usage/plotting
write.csv(plan.day, file.path("./data",'weekdaydemand.csv'),row.names = FALSE)
write.csv(plan.end,file.path("./data",'weekenddemand.csv'),row.names = FALSE)
write.csv(error.day,file.path("./data",'weeklyerror.csv'),row.names = FALSE)
write.csv(error.end,file.path("./data",'weekenderror.csv'),row.names = FALSE)
write.csv(anova.d,file.path("./data",'anova.csv'),row.names = FALSE)

