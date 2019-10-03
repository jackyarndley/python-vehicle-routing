#Setting current working directory to source file location
rstudioapi::getActiveDocumentContext
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#Loading required libraries
library(tidyverse)
library(ggplot2)

#Reading in data and cleaning
demand = read_csv("demandData.csv", col_names=TRUE)
demand = gather(demand,key=date,value=demand,ends_with("19"))
demand$weekday = as.factor(weekdays(as.Date(demand$date,format="%d/%m/%Y")))
demand$Type <- sub("(\\w+).*", "\\1", demand$Supermarket)

weekday <- filter(demand,weekday%in% c("Monday","Tuesday","Wednesday","Thursday","Friday"))
weekday.g <- group_by(weekday,Supermarket)
d.error.day <- summarise(weekday.g,sum = sum(demand))
d.plan.day <- summarise(weekday.g,demand = ceiling(mean(demand)))
weekend <- filter(demand,weekday == "Saturday")
weekend.g <- group_by(weekend,Supermarket)
d.error.end <- summarise(weekend.g, sum = sum(demand))
d.plan.end <- summarise(weekend.g,demand = ceiling(mean(demand)))

d.error.day$demand <- d.plan.day$demand
d.error.day$error <- d.error.day$demand*20/d.error.day$sum

d.error.end$demand <- d.plan.end$demand
d.error.end$error <- d.error.end$demand*4/d.error.end$sum

write.csv(d.plan.day,'weekdaydemand.csv',row.names = FALSE)
write.csv(d.plan.end,'weekenddemand.csv',row.names = FALSE)
write.csv(d.plan.day,'weeklyerror.csv',row.names = FALSE)
write.csv(d.plan.end,'weekenderror.csv',row.names = FALSE)



