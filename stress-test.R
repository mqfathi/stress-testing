library(igraph)
library(readxl)
library(visNetwork)
library(ggplot2)
library(ggrepel)

OrderList <- read_excel("C:/Users/L03128674/Downloads/Supply chain logisitcs problem.xlsx", sheet = "OrderList")
FreightRates <- read_excel("C:/Users/L03128674/Downloads/Supply chain logisitcs problem.xlsx", sheet = "FreightRates")
PlantsPorts <- read_excel("C:/Users/L03128674/Downloads/Supply chain logisitcs problem.xlsx", sheet = "PlantPorts")
WhCapacities <- read_excel("C:/Users/L03128674/Downloads/Supply chain logisitcs problem.xlsx", sheet = "WhCapacities")
ProductsPerPlant <- read_excel("C:/Users/L03128674/Downloads/Supply chain logisitcs problem.xlsx", sheet = "ProductsPerPlant")
WhCosts <- read_excel("C:/Users/L03128674/Downloads/Supply chain logisitcs problem.xlsx", sheet = "WhCosts")

# Create a data frame with two columns of nominal information
df <- data.frame(
  Plant = PlantsPorts$PlantCode,
  Port = PlantsPorts$Port
)






#vulnerability plant based demand
demand = data.frame(table(OrderList$PlantCode))
# Find the missing labels in dataset2
missing_labels <- WhCapacities$PlantID[!unique(WhCapacities$PlantID) %in% unique(OrderList$PlantCode)]
# Add the missing labels to dataset2 and assign them a value of zero
for (label in missing_labels) {
  demand <- rbind(demand, data.frame(Var1 = label, Freq = 0))
}
demand = demand[match(WhCapacities$PlantID, demand$Var1), ]
ind_vulnerability_demand = -(demand$Freq/sum(demand$Freq)*log(demand$Freq/sum(demand$Freq)))
na_id = is.na(ind_vulnerability_demand)
ind_vulnerability_demand[na_id]=0
total_vulnerability_capacity = -sum(WhCapacities$DailyCapacity/sum(WhCapacities$DailyCapacity)*log(WhCapacities$DailyCapacity/sum(WhCapacities$DailyCapacity)))
Vulnerability = (total_vulnerability_capacity-ind_vulnerability_demand)/total_vulnerability_capacity
products=table(ProductsPerPlant$PlantCode)

vulnerability_VP = Vulnerability
H_VP =total_vulnerability_capacity

df <- data.frame(x = WhCapacities$DailyCapacity,
                   y = Vulnerability,
                   label = WhCapacities$PlantID, 
                 size =table(ProductsPerPlant$PlantCode))

total_vulnerability_i = vector()
for (i in 1:length(WhCapacities$PlantID)){
  total_vulnerability_i[i]= -sum(WhCapacities$DailyCapacity[-i]/sum(WhCapacities$DailyCapacity[-i])*log(WhCapacities$DailyCapacity[-i]/sum(WhCapacities$DailyCapacity[-i])))
}

Global_VP = total_vulnerability_capacity/total_vulnerability_i*vulnerability_VP

Demand = df$size.Freq


plot = ggplot(df, aes(x, y)) +
  geom_point(aes(size = Demand)) +
  labs(title = expression(paste(v[p]))) +
  theme(plot.title = element_text(size = 28, face = "bold"))


VP= plot + geom_text_repel(aes(label = label),size = 4,max.overlaps=Inf) +
  xlab("Capacity") + ylab("Vulnerability")+
  theme(
    legend.title = element_text(size = 14), # Adjust size as needed for legend title
    legend.text = element_text(size = 12), # Adjust size as needed for legend text
    axis.title = element_text(size = 16), # Adjust size as needed for axis titles
    axis.text = element_text(size = 14) # Adjust size as needed for axis text (x and y)
  )
VP
#####################
#vulnerability plant based cost

demand = data.frame(table(OrderList$PlantCode))
Unites = aggregate(Unitquantity~PlantCode,OrderList,sum)
# Find the missing labels in dataset2
missing_labels <- WhCapacities$PlantID[!unique(WhCapacities$PlantID) %in% unique(OrderList$PlantCode)]
# Add the missing labels to dataset2 and assign them a value of zero
for (label in missing_labels) {
  demand <- rbind(demand, data.frame(Var1 = label, Freq = 0))
  Unites <- rbind(Unites, data.frame(PlantCode = label, Unitquantity = 0))
}
demand = demand[match(WhCapacities$PlantID, demand$Var1), ]
Unites = Unites[match(WhCapacities$PlantID, Unites$PlantCode), ]
ind_vulnerability_demand = -(WhCosts$Costunit*Unites$Unitquantity)/sum((WhCosts$Costunit*Unites$Unitquantity))*log((WhCosts$Costunit*Unites$Unitquantity)/sum((WhCosts$Costunit*Unites$Unitquantity)))

na_id = is.na(ind_vulnerability_demand)
ind_vulnerability_demand[na_id]=0

total_vulnerability_capacity = -sum((WhCosts$Costunit*WhCapacities$DailyCapacity)*(WhCapacities$DailyCapacity/sum((WhCosts$Costunit*WhCapacities$DailyCapacity)*WhCapacities$DailyCapacity)*log((WhCosts$Costunit*WhCapacities$DailyCapacity)*WhCapacities$DailyCapacity/sum((WhCosts$Costunit*WhCapacities$DailyCapacity)*WhCapacities$DailyCapacity))))

Vulnerability = (total_vulnerability_capacity-ind_vulnerability_demand)/total_vulnerability_capacity
products=table(ProductsPerPlant$PlantCode)

vulnerability_VWP = Vulnerability
H_VWP =total_vulnerability_capacity

#plotting
df <- data.frame(x = WhCapacities$DailyCapacity,
                 y = Vulnerability,
                 label = WhCapacities$PlantID, 
                 size =table(WhCosts$Costunit))

total_vulnerability_i = vector()
for (i in 1:length(WhCapacities$PlantID)){
  total_vulnerability_i[i]= -sum((WhCosts$Costunit[-i]*mean(OrderList$Unitquantity[-i])*WhCapacities$DailyCapacity[-i])*(WhCapacities$DailyCapacity[-i]/sum((WhCosts$Costunit[-i]*mean(OrderList$Unitquantity[-i])*WhCapacities$DailyCapacity[-i])*WhCapacities$DailyCapacity[-i])*log((WhCosts$Costunit[-i]*mean(OrderList$Unitquantity[-i])*WhCapacities$DailyCapacity[-i])*WhCapacities$DailyCapacity[-i]/sum((WhCosts$Costunit[-i]*mean(OrderList$Unitquantity[-i])*WhCapacities$DailyCapacity[-i])*WhCapacities$DailyCapacity[-i]))))
}

Global_VWP = total_vulnerability_capacity/total_vulnerability_i*vulnerability_VWP

Unit_Cost = WhCosts$Costunit

plot = ggplot(df, aes(x , y )) +
  geom_point(aes(size=Unit_Cost)) +
  labs(title = expression(paste(v[W[p]]))) +
  theme(plot.title = element_text(size = 28, face = "bold"))

VWP = plot + geom_text_repel(aes(label = label),size = 4.5,max.overlaps=Inf) +
  xlab("Capacity") + ylab("Vulnerability")+
  theme(
    legend.title = element_text(size = 14), # Adjust size as needed for legend title
    legend.text = element_text(size = 12), # Adjust size as needed for legend text
    axis.title = element_text(size = 16), # Adjust size as needed for axis titles
    axis.text = element_text(size = 14) # Adjust size as needed for axis text (x and y)
  )

VWP

###########
#vulnerablity port cost

A = aggregate(cbind(minimumcost , rate) ~ Carrier+ orig_port_cd+ minm_wgh_qty+ max_wgh_qty+ svc_cd+  tpt_day_cnt,FreightRates, mean)
planed_cost = A$minimumcost+A$rate*(A$minm_wgh_qty+A$max_wgh_qty)/2
A = cbind(A, planed_cost)
C = merge(OrderList, A, by.x = c("Carrier", "OriginPort", "ServiceLevel", "TPT"), by.y = c("Carrier", "orig_port_cd", "svc_cd", "tpt_day_cnt"))
B = subset(C, C[, "Weight"] >= C[, "minm_wgh_qty"] & C[, "Weight"] <= C[, "max_wgh_qty"])
actual_cost = B$minimumcost + B$rate*B$Weight
B = cbind(B, actual_cost)


transcost_actual = aggregate(actual_cost~OriginPort,B,sum)
transcost_planned= aggregate(planed_cost~orig_port_cd,A,sum)

missing_labels <- transcost_planned$orig_port_cd[!unique(transcost_planned$orig_port_cd) %in% unique(transcost_actual$OriginPort)]
# Add the missing labels to dataset2 and assign them a value of zero
for (label in missing_labels) {
  transcost_actual <- rbind(transcost_actual, data.frame(OriginPort = label, actual_cost = 0))
}
transcost_actual = transcost_actual[match(transcost_planned$orig_port_cd, transcost_actual$OriginPort), ]


ind_vulnerability_transcost = -transcost_actual$actual_cost/sum(transcost_actual$actual_cost)*log(transcost_actual$actual_cost/sum(transcost_actual$actual_cost))
na_id = is.na(ind_vulnerability_transcost)
ind_vulnerability_transcost[na_id]=0

total_vulnerability_transcost = -sum(transcost_planned$planed_cost/sum(transcost_planned$planed_cost)*log(transcost_planned$planed_cost/sum(transcost_planned$planed_cost)))
Vulnerability = ((total_vulnerability_transcost-ind_vulnerability_transcost)/total_vulnerability_transcost)
products=table(ProductsPerPlant$PlantCode)

#plotting
Label = aggregate(Weight~OriginPort, OrderList,mean)
missing_labels <- transcost_planned$orig_port_cd[!unique(transcost_planned$orig_port_cd) %in% unique(Label$OriginPort)]
# Add the missing labels to dataset2 and assign them a value of zero
for (label in missing_labels) {
  Label <- rbind(Label, data.frame(OriginPort = label, Weight = 1))
}
Label_ct = Label[match(transcost_planned$orig_port_cd, Label$OriginPort), ]


vulnerability_VCT = Vulnerability
H_VCT =total_vulnerability_transcost

df <- data.frame(x = transcost_planned$planed_cost,
                 y = Vulnerability,
                 label = transcost_planned$orig_port_cd, 
                 size =Label_ct$Weight)
Average_weight = Label_ct$Weight
plot = ggplot(df, aes(x , y )) +
  geom_point(aes(size=Average_weight)) +
  labs(title = expression(paste(v[C[T]]))) + 
  theme(plot.title = element_text(size = 28, face = "bold"))

VCT = plot + geom_text_repel(aes(label = label),size = 4,max.overlaps=Inf) + xlab("Transport Cost") + ylab("Vulnerability")+
  theme(
    legend.title = element_text(size = 14), # Adjust size as needed for legend title
    legend.text = element_text(size = 12), # Adjust size as needed for legend text
    axis.title = element_text(size = 16), # Adjust size as needed for axis titles
    axis.text = element_text(size = 14) # Adjust size as needed for axis text (x and y)
  )
VCT


total_vulnerability_i = vector()
for (i in 1:length(transcost_planned$orig_port_cd)){
  total_vulnerability_i[i]= -sum(transcost_planned$planed_cost[-i]/sum(transcost_planned$planed_cost[-i])*log(transcost_planned$planed_cost[-i]/sum(transcost_planned$planed_cost[-i])))
}

Global_VCT = total_vulnerability_transcost/total_vulnerability_i*vulnerability_VCT

#################
#vulnerability transport time
Time = aggregate(tpt_day_cnt~orig_port_cd, FreightRates,mean)
Time_actual =aggregate(OrderList$Shipaheaddaycount+OrderList$ShipLateDaycount~OriginPort, OrderList,mean) 
# Find the missing labels in dataset2
missing_labels <- Time$orig_port_cd[!unique(Time$orig_port_cd) %in% unique(Time_actual$OriginPort)]
# Add the missing labels to dataset2 and assign them a value of zero
colnames(Time_actual) = c("OriginPort", "time")
for (label in missing_labels) {
  Time_actual <- rbind(Time_actual, data.frame(OriginPort = label, time = 0))
}
Time_actual = Time_actual[match(Time$orig_port_cd, Time_actual$OriginPort), ]

ind_vulnerability_time = -Time_actual$time/sum(Time_actual$time)*log(Time_actual$time/sum(Time_actual$time))
na_id = is.na(ind_vulnerability_time)
ind_vulnerability_time[na_id]=0

total_vulnerability_time = -sum(Time$tpt_day_cnt/sum(Time$tpt_day_cnt)*log(Time$tpt_day_cnt/sum(Time$tpt_day_cnt)))
Vulnerability = (total_vulnerability_time-ind_vulnerability_time)/total_vulnerability_time



Label = aggregate(Unitquantity~OriginPort, OrderList,mean)
missing_labels <- transcost_planned$orig_port_cd[!unique(transcost_planned$orig_port_cd) %in% unique(Label$OriginPort)]
# Add the missing labels to dataset2 and assign them a value of zero
for (label in missing_labels) {
  Label <- rbind(Label, data.frame(OriginPort = label, Unitquantity = 1))
}
Label = Label[match(transcost_planned$orig_port_cd, Label$OriginPort), ]


vulnerability_VT = Vulnerability
H_VT =total_vulnerability_time

df <- data.frame(x = Time$tpt_day_cnt,
                 y = Vulnerability,
                 label = Time$orig_port_cd, 
                 size =Label$Unitquantity)

Unit_quantity = Label$Unitquantity

plot = ggplot(df, aes(x , y )) +
  geom_point(aes(size=Unit_quantity)) +
  labs(title = expression(paste(v[T]))) + 
  theme(plot.title = element_text(size = 28, face = "bold"))

VT =plot + geom_text_repel(aes(label = label),size = 3,max.overlaps=Inf) +
  xlab("Transit time") + ylab("Vulnerability")
VT


total_vulnerability_i = vector()
for (i in 1:length(transcost_planned$orig_port_cd)){
  total_vulnerability_i[i]= -sum(Time$tpt_day_cnt[-i]/sum(Time$tpt_day_cnt[-i])*log(Time$tpt_day_cnt[-i]/sum(Time$tpt_day_cnt[-i])))
}

Global_VCT = total_vulnerability_time/total_vulnerability_i*vulnerability_VT



# Install and load the gridExtra package
install.packages("gridExtra")
library(gridExtra)

# Arrange the plots in a 2x2 grid
grid.arrange(VP, VWP, VCT, VT, ncol = 2)


##############################
Impact_VP = ifelse(vulnerability_VP < 1, vulnerability_VP * H_VP, 0)#(vulnerability_VP) * H_VP
Impact_VWP = ifelse(vulnerability_VWP < 1, vulnerability_VWP * H_VWP, 0)# (1-vulnerability_VWP) * H_VWP
Impact_VCT = ifelse(vulnerability_VCT < 1, vulnerability_VCT * H_VCT, 0)#(1-vulnerability_VCT) * H_VCT
Impact_VT = ifelse(vulnerability_VT < 1, vulnerability_VT * H_VT, 0)#(1-vulnerability_VT) * H_VT

Impact_VP = -(vulnerability_VP) * (WhCapacities$DailyCapacity/sum(WhCapacities$DailyCapacity)*log(WhCapacities$DailyCapacity/sum(WhCapacities$DailyCapacity)))
Impact_VWP = (vulnerability_VWP) * H_VWP
Impact_VCT = (vulnerability_VCT) * H_VCT
Impact_VT = (vulnerability_VT) * H_VT



df <- data.frame(x = vulnerability_VP,
                 y = Impact_VP,
                 label = WhCapacities$PlantID, 
                 size =table(ProductsPerPlant$PlantCode))
#df <- subset(df, x < 1)
Demand = df$size.Freq

plot = ggplot(df, aes(x , y )) +
  geom_point(aes(size=Demand)) +
  labs(title = expression(paste(I[p]))) + 
  theme(plot.title = element_text(size = 28, face = "bold"))

IVP = plot + geom_text_repel(aes(label = label),size = 3,max.overlaps=Inf) +
  xlab("Vulnerability") + ylab("Impact")

###
df <- data.frame(x = vulnerability_VWP,
                 y = Impact_VWP,
                 label = WhCapacities$PlantID, 
                 size =table(WhCosts$Costunit))
#df <- subset(df, x < 1)
Unit_Cost = WhCosts$Costunit

plot = ggplot(df, aes(x , y )) +
  geom_point(aes(size=Unit_Cost)) +
  labs(title = expression(paste(I[W[p]]))) + 
  theme(plot.title = element_text(size = 28, face = "bold"))

IVWP=plot + geom_text_repel(aes(label = label),size = 3,max.overlaps=Inf) +
  xlab("Vulnerability") + ylab("Impact")


#####
df <- data.frame(x = vulnerability_VCT,
                 y = Impact_VCT,
                 label = transcost_planned$orig_port_cd, 
                 size =Label_ct$Weight)
#df <- subset(df, x < 1)
Average_weight = Label_ct$Weight
plot = ggplot(df, aes(x , y )) +
  geom_point(aes(size=Average_weight)) +
  labs(title = expression(paste(I[C[T]]))) + 
  theme(plot.title = element_text(size = 28, face = "bold"))

IVCT= plot + geom_text_repel(aes(label = label),size = 3,max.overlaps=Inf) + xlab("Vulnerability") + ylab("Impact")

#####
df <- data.frame(x = vulnerability_VT,
                 y = Impact_VT,
                 label = Time$orig_port_cd, 
                 size =Label$Unitquantity)
#df <- subset(df, x < 1)

Unit_quantity = Label$Unitquantity

plot = ggplot(df, aes(x , y )) +
  geom_point(aes(size=Unit_quantity)) +
  labs(title = expression(paste(I[T]))) + 
  theme(plot.title = element_text(size = 28, face = "bold"))

IVT= plot + geom_text_repel(aes(label = label),size = 3,max.overlaps=Inf) +
  xlab("Vulnerability") + ylab("Impact")



grid.arrange(IVP, IVWP, IVCT, IVT, ncol = 2)
########################################
par(mfrow = c(2, 2))
alpha = 0.05
vulnerability_VP <- ifelse(vulnerability_VP == 1, 0, vulnerability_VP)

density_vp <- density(vulnerability_VP[vulnerability_VP > 0], adjust = 1)
percentile_95 <- quantile(vulnerability_VP[vulnerability_VP > 0], 1-alpha)

plot(density_vp, main = expression(paste(V[P])), xlab = "Vulnerability", lwd = 2)
polygon(c(density_vp$x[density_vp$x > percentile_95][1],density_vp$x[density_vp$x > percentile_95], density_vp$x[density_vp$x > percentile_95][1]), c(0, density_vp$y[density_vp$x > percentile_95], 0), col = "red")

mean_area <- mean(vulnerability_VP[vulnerability_VP > percentile_95])
abline(v = mean_area, col = "blue", lty = 2,  lwd = 2)
abline(v = percentile_95, col = "black", lty = 2, lwd = 2)
text(percentile_95, max(density_vp$y), "VaR" , pos = 2)
text(mean_area, max(density_vp$y),"CVaR", pos = 4)

############
vulnerability_VWP <- ifelse(vulnerability_VWP == 1, 0, vulnerability_VWP)

density_vwp <- density(vulnerability_VWP[vulnerability_VWP > 0])
percentile_95 <- quantile(vulnerability_VWP[vulnerability_VWP > 0], 1-alpha)

plot(density_vwp, main = expression(paste(V[W[P]])), xlab = "Vulnerability", lwd = 2)
polygon(c(density_vwp$x[density_vwp$x > percentile_95][1],density_vwp$x[density_vwp$x > percentile_95], density_vwp$x[density_vwp$x > percentile_95][1]), c(0, density_vwp$y[density_vwp$x > percentile_95], 0), col = "red")

mean_area <- mean(vulnerability_VWP[vulnerability_VWP > percentile_95])
#mean_area <- quantile(density_vwp$x[density_vwp$x > percentile_95], 0.5)
#mean_area <- mean(density_vwp$x[density_vwp$x > percentile_95]*density_vwp$y[density_vwp$x > percentile_95])
LINE = sort(density_vwp$x[density_vwp$y > mean_area], decreasing = T)[1]

abline(v = mean_area, col = "blue", lty = 2,  lwd = 2)
abline(v = percentile_95, col = "black", lty = 2, lwd = 2)
text(percentile_95, max(density_vwp$y), "VaR", pos = 2)
text(mean_area, max(density_vwp$y), "CVaR", pos = 4)



#####
vulnerability_VCT <- ifelse(vulnerability_VCT == 1, 0, vulnerability_VCT)

density_vct <- density(vulnerability_VCT[vulnerability_VCT > 0])
percentile_95 <- quantile(vulnerability_VCT[vulnerability_VCT > 0], 0.95)

plot(density_vct, main = expression(paste(V[C[T]])), xlab = "Vulnerability", lwd = 2)
polygon(c(density_vct$x[density_vct$x > percentile_95][1],density_vct$x[density_vct$x > percentile_95], density_vct$x[density_vct$x > percentile_95][1]), c(0, density_vct$y[density_vct$x > percentile_95], 0), col = "red")

mean_area <- mean(vulnerability_VCT[vulnerability_VCT > percentile_95])
abline(v = mean_area, col = "blue", lty = 2,  lwd = 2)
abline(v = percentile_95, col = "black", lty = 2, lwd = 2)
text(percentile_95, max(density_vct$y), "VaR", pos = 2)
text(mean_area, max(density_vct$y), "CVaR", pos = 4)

#####################

# Calcular la densidad de la variable "vulnerability_VT"
vulnerability_VT <- ifelse(vulnerability_VT == 1, 0, vulnerability_VT)

density_vt <- density(vulnerability_VT[vulnerability_VT > 0])
percentile_95 <- quantile(vulnerability_VT[vulnerability_VT > 0], 0.95)

plot(density_vt, main = expression(paste(V[T])), xlab = "Vulnerability", lwd = 2)
polygon(c(density_vt$x[density_vt$x > percentile_95][1],density_vt$x[density_vt$x > percentile_95],density_vt$x[density_vt$x > percentile_95][1]), c(0,density_vt$y[density_vt$x > percentile_95],0), col = "red")
mean_area <- mean(vulnerability_VT[vulnerability_VT >= percentile_95])
abline(v = mean_area, col = "blue", lty = 2,  lwd = 2)
abline(v = percentile_95, col = "black", lty = 2, lwd = 2)
text(percentile_95, max(density_vt$y), "VaR", pos = 2)
text(mean_area, max(density_vt$y), "CVaR", pos = 4)

