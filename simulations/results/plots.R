#----------------------------
# Load libraries
#----------------------------
library(dplyr)
library(ggplot2)

#----------------------------
# False positive rate (FPR)
#----------------------------
# Function that reads file with results and create a plot using ggplot2
df2plot <- function(file, ylab){
  # Read files
  df = read.csv2(file, sep=",") %>%
           mutate(fpr = 100*as.numeric(fpr), mean = as.numeric(mean)) %>%
           mutate(h2g = ifelse(scenario == 1, 0.3, 0.5)) %>%
           mutate(h2d = ifelse(scenario == 3, 0.3, 0)) %>%
           mutate(h2b = ifelse(scenario == 1, 0.6, ifelse(scenario == 2, 0.3, 0)))
  
  # Change method names
  for (method in c("pglmm", "glmnet", "glmnetPC", "ggmix")){
    df[df$method == paste0(method,"FPR"), "method"] <- method
  }
  
  # ggplot
  ggplot(df,aes(x=fpr,y=mean, linetype = method, color = method))+
    geom_line(size = 0.75)+
    facet_grid(K~scenario, labeller=label_bquote(cols=scenario:.(scenario),rows=K==.(K)))+
    #facet_grid(K~scenario+h2g+h2d+h2b, labeller=label_bquote(cols=list(h[g]^2,h[b]^2,h[d]^2)==list(.(h2g),.(h2b),.(h2d)),rows=K==.(K)))+
    labs(x="False positive rate (FPR) %",y=ylab)+
    scale_color_brewer(palette = "Spectral") +
    labs(color  = "method", linetype = "method")+
    theme_bw() + 
    theme(plot.background = element_blank(),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank() )+
    #theme(panel.border= element_blank())+
    theme(axis.line.x = element_line(color="black", size = 0.5),
          axis.line.y = element_line(color="black", size = 0.5))
}

# AUC
df2plot(file="fpr_auc.csv", ylab="AUC")

# Bias
df2plot(file="fpr_bias.csv", ylab="Relative bias (%)")

# RMSE
df2plot(file="fpr_rmse.csv", ylab="Root mean squared error (RMSE)")

# TPR
df2plot(file="fpr_tpr.csv", ylab="True positive rate (TPR)")

#----------------------------
# Model selection
#----------------------------

# Function that reads file with results and create a plot using ggplot2
df2boxplot <- function(file, ylab, methods){
  # Read files
  df = read.csv2(file, sep=",") %>%
       mutate(value = as.numeric(value))
  
  if (!missing(methods)){
    df = filter(df, method %in% methods)
  }
  
  if (file == "model_size.csv"){
    df = mutate(df, value = log10(value+1))
  }
  
  # Change method names
  for (method in c("cv_glmnet", "cv_glmnetPC")){
    df[df$method == method, "method"] <- substr(method, 4, nchar(method))
  }
  
  # ggplot
  ggplot(df,aes(x=method,y=value,color = method))+
    geom_boxplot()+
    facet_grid(K~scenario, labeller=label_bquote(cols=scenario==.(scenario),rows=K==.(K)))+
    labs(x="Method",y=ylab)+
    scale_color_brewer(palette = "Set1") +
    theme_bw() + 
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank())+
    theme(plot.background = element_blank(),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank() )+
    #theme(panel.border= element_blank())+
    theme(axis.line.x = element_line(color="black", size = 0.5),
          axis.line.y = element_line(color="black", size = 0.5))
}

# Model size
df2boxplot(file="model_size.csv", ylab=expression(paste("Model size (log"[10], " scale)", sep="")), methods = c("pglmmAIC", "pglmmBIC", "ggmixBIC", "cv_glmnetPC"))

# AUC
df2boxplot(file="model_auc.csv", ylab="AUC", methods = c("pglmmAIC", "pglmmBIC", "ggmixBIC", "cv_glmnetPC"))

# Bias
df2boxplot(file="model_bias.csv", ylab="Relative bias (%)",  methods = c("pglmmAIC", "pglmmBIC", "ggmixBIC", "cv_glmnetPC"))

# RMSE
df2boxplot(file="model_rmse.csv", ylab="Root mean squared error (RMSE)",  methods = c("pglmmAIC", "pglmmBIC", "ggmixBIC", "cv_glmnetPC"))

# TPR
df2boxplot(file="model_tpr.csv", ylab="True positive rate (TPR)",  methods = c("pglmmAIC", "pglmmBIC", "ggmixBIC", "cv_glmnetPC"))
