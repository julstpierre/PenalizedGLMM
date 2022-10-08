#----------------------------
# Load libraries
#----------------------------
library(dplyr)
library(ggplot2)
library(ggpubr)

#----------------------------
#Model size
#----------------------------
# Function that reads file with results and create a plot using ggplot2
size2plot <- function(file, ylab, geo_ = NULL, K_ = NULL, methods = NULL, kin_ = "NONE"){
  # Read files
  if (!is.null(K_)){
    df <- read.csv2(file, sep=",") %>% filter(K %in% K_) %>% filter(kin %in% kin_)
  } else {
    df <- read.csv2(file, sep=",") %>% filter(kin %in% kin_)
  }
  
  if (!is.null(geo_)){
    if (file == "size_auc_all.csv"){
        df <- filter(df, geo %in% geo_) %>%
        mutate(size = as.numeric(size), auc = as.numeric(auc)) %>%
        mutate(geo = ifelse(geo == "1d", "1d linear admixture", ifelse(geo == "ind", "independent", "related")))
    } else {
        df <- filter(df, geo %in% geo_) %>%
        mutate(size = as.numeric(size), mean = as.numeric(mean)) %>%
        mutate(geo = ifelse(geo == "1d", "1d linear admixture", ifelse(geo == "ind", "independent", "related")))
    }
  } else{
    df <- mutate(size = as.numeric(size), mean = as.numeric(mean)) %>%
      mutate(geo = ifelse(geo == "1d", "1d linear admixture", ifelse(geo == "ind", "independent", "related")))
  }
  
  # Change method names
  for (method in c("pglmm", "glmnet", "glmnetPC", "ggmix")){
    df[df$method == paste0(method,"size"), "method"] <- method
  }
  
  if (!is.null(methods)){
    df = filter(df, method %in% methods)
  }
  
  # ggplot
  if (length(unique(df$geo)) > 1){
    if (file == "size_auc_all.csv"){
      ggplot(filter(df, size %in% c(10, 20, 30, 40, 50)),aes(x=as.character(size),y=auc, color = method))+
        geom_boxplot()+
        facet_grid(K~geo, labeller=label_bquote(cols=.(geo),rows=K==.(K)))+
        labs(x="Model size",y=ylab)+
        scale_color_brewer(palette = "Set1") +
        labs(color  = "method", linetype = "method")+
        theme_bw() + 
        theme(legend.position="bottom")+
        theme(plot.background = element_blank(),
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank() )+
        #theme(panel.border= element_blank())+
        theme(axis.line.x = element_line(color="black", size = 0.5),
              axis.line.y = element_line(color="black", size = 0.5))
    } else {
      ggplot(df,aes(x=size,y=mean, linetype = method, color = method))+
        geom_line(size = 0.75)+
        facet_grid(K~geo, labeller=label_bquote(cols=.(geo),rows=K==.(K)))+
        #facet_grid(K~scenario+h2g+h2b, labeller=label_bquote(cols=list(h[g]^2,h[b]^2)==list(.(h2g),.(h2b)),rows=K==.(K)))+
        labs(x="Model size",y=ylab)+
        scale_color_brewer(palette = "Spectral") +
        labs(color  = "method", linetype = "method")+
        theme_bw() + 
        theme(legend.position="bottom")+
        theme(plot.background = element_blank(),
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank() )+
        #theme(panel.border= element_blank())+
        theme(axis.line.x = element_line(color="black", size = 0.5),
              axis.line.y = element_line(color="black", size = 0.5))
    }
  } else {
    ggplot(df,aes(x=size,y=mean, linetype = method, color = method))+
      geom_line(size = 0.75)+
      labs(x="Model size",y=ylab)+
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
}

#------------------------
# Simulated genotypes
#------------------------
# AUC
#fpr2plot(file="fpr_auc.csv", ylab="AUC", geo = c("1d", "ind"))
size2plot(file="size_auc.csv", ylab="AUC", geo = c("1d", "ind"), K = c(10, 20))
size2plot(file="size_auc_all.csv", ylab="AUC", geo = c("1d", "ind"), K = c(10, 20), methods = c("glmnetPC", "pglmm"))

# Bias
#fpr2plot(file="fpr_bias.csv", ylab="Relative bias (%)", geo = c("1d", "ind"))
#size2plot(file="size_bias.csv", ylab="Relative bias (%)", geo = c("1d", "ind"))

# RMSE
#fpr2plot(file="fpr_rmse.csv", ylab=expression(paste("RMSE (", hat(beta), ")")), geo = c("1d", "ind"))
size2plot(file="size_rmse.csv", ylab=expression(paste("RMSE (", hat(beta), ")")), geo = c("1d", "ind"), K = c(10, 20))

# TPR
#fpr2plot(file="fpr_tpr.csv", ylab="TPR", geo = c("1d", "ind"))
size2plot(file="size_tpr.csv", ylab="TPR", geo = c("1d", "ind"), K = c(10, 20))

#------------------------------
# Related individuals from UKBB
#------------------------------
kin = "ALL"

# AUC
p1 <- size2plot(file="size_auc.csv", ylab="AUC", geo = c("UKBB"), kin_ = kin)

# Bias
#size2plot(file="size_bias.csv", ylab="Relative bias (%)", geo = c("UKBB"))

# RMSE
p2 <- size2plot(file="size_rmse.csv", ylab=expression(paste("RMSE (", hat(beta), ")")), geo = c("UKBB"), kin_ = kin)

# TPR
p3 <- size2plot(file="size_tpr.csv", ylab="TPR", geo = c("UKBB"), kin_ = kin)

ggarrange(p1, p2, p3, common.legend = TRUE, legend = "bottom", ncol = 2, nrow = 2)

#----------------------------
# Model selection
#----------------------------

# Function that reads file with results and create a plot using ggplot2
df2boxplot <- function(file, ylab, methods, geo_ = NULL, K_ = NULL, kin_ = "NONE", logscale = TRUE, ymin = NA, ymax = NA, change_labels = TRUE, print_table = FALSE){
  # Read files
  if (!is.null(K_)){
    df <- read.csv2(file, sep=",") %>% 
            filter(K %in% K_) %>%
            filter(kin %in% kin_) %>%
            mutate(value = as.numeric(value))
  } else {
    df <- read.csv2(file, sep=",") %>%
          filter(kin %in% kin_) %>%
            mutate(value = as.numeric(value))
  }

  if (!is.null(geo_)){
    df <- filter(df, geo %in% geo_)
  } 
  
  df <- mutate(df, geo = ifelse(geo == "1d", "1d linear admixture", ifelse(geo == "ind", "independent", "related")))
  
  if (!missing(methods)){
    df = filter(df, method %in% methods)
  }
  
  if (file == "model_size.csv" & logscale == TRUE){
    df = mutate(df, value = log10(value+1))
  }
  
  if (change_labels == TRUE){
    df[df$method == "ggmixBIC", "method"] <- "ggmix (BIC)"
    #df[df$method == "ggmix", "method"] <- "ggmix (CV)"
    df[df$method == "cv_glmnetPC", "method"] <- "glmnetPC (10-fold CV)"
    #df[df$method == "glmnetPC", "method"] <- "glmnetPC (CV)"
    #df[df$method == "glmnet", "method"] <- "glmnet (CV)"
    df[df$method == "pglmmAIC", "method"] <- "pglmm (AIC)"
    df[df$method == "pglmmBIC", "method"] <- "standard gBLUP"
    df[df$method == "pglmm", "method"] <- "pglmm (CV)"
  }
  
  # ggplot
  if (print_table){
    group_by(df, method) %>%
      summarise(mean = mean(value), 
                sd = sd(value),
                median = median(value), 
                IQR = IQR(value))
  } else if (length(unique(df$geo)) > 1){
    ggplot(df,aes(x=method,y=value,color = method))+
      geom_boxplot(outlier.shape=)+
      facet_grid(K~geo, labeller=label_bquote(cols=.(geo),rows=K==.(K)))+
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
  } else{
      if (!all(is.na(c(ymax,ymin)))){
      ggplot(df,aes(x=method,y=value,color = method))+
        geom_boxplot()+
        ylim(ymin,ymax)+
        labs(x="Method",y=ylab)+
        scale_color_brewer(palette = "Set2") +
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
      } else {
        ggplot(df,aes(x=method,y=value,color = method))+
          geom_boxplot()+
          labs(x="Method",y=ylab)+
          scale_color_brewer(palette = "Set2") +
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
  }
  
}

#----------------------------
# Related subjects from UKBB
#----------------------------

kin = "ALL"

# Model size
q1_ <- df2boxplot(file="model_size.csv", ylab=expression(paste("Model size (log"[10], " scale)", sep="")), methods = c("pglmm", "pglmmAIC", "glmnet","glmnetPC", "ggmix"), geo = c("UKBB"), kin_ = kin, change_labels = TRUE)

# AUC
q2_ <- df2boxplot(file="model_auc.csv", ylab="AUC", methods = c("pglmm", "pglmmAIC", "glmnet", "glmnetPC", "ggmix", "pglmmBIC"), geo = c("UKBB"), kin_ = kin, change_labels = TRUE)

# RMSE
q3_ <- df2boxplot(file="model_rmse.csv", ylab=expression(paste("RMSE (", hat(beta), ")")), methods = c("pglmm", "pglmmAIC", "glmnet", "glmnetPC", "ggmix"), geo = c("UKBB"), kin_ = kin, change_labels = TRUE, ymin = 0.025, ymax = 0.04)

# TPR
q4_ <- df2boxplot(file="model_tpr.csv", ylab="TPR", methods = c("pglmm", "pglmmAIC", "glmnet", "glmnetPC","ggmix"), geo = c("UKBB"), kin_ = kin, change_labels = TRUE)

# Precision
q5_ <- df2boxplot(file="model_ppv.csv", ylab="Precision", methods = c("pglmm", "pglmmAIC", "glmnet", "glmnetPC", "ggmix"), geo = c("UKBB"), kin_ = kin, change_labels = TRUE, ymin=0, ymax=0.4)

ggarrange(q2_, q1_, q3_, q4_, q5_, common.legend = TRUE, legend = "bottom", ncol = 2, nrow = 3)

# Tables 
# Model size
df2boxplot(file="model_size.csv", methods = c("pglmm", "pglmmAIC", "glmnet", "glmnetPC","ggmix"), geo = c("UKBB"), kin_ = kin, change_labels = TRUE, print_table = TRUE, logscale = FALSE)

# AUC
df2boxplot(file="model_auc.csv", ylab="AUC", methods = c("pglmm", "pglmmAIC", "glmnet", "glmnetPC","ggmix", "pglmmAIC"), geo = c("UKBB"), kin_ = kin, change_labels = TRUE, print_table = TRUE)

# RMSE
df2boxplot(file="model_rmse.csv", ylab=expression(paste("RMSE (", hat(beta), ")")), methods = c("pglmm", "pglmmAIC", "glmnet", "glmnetPC","ggmix"), geo = c("UKBB"), kin_ = kin, change_labels = TRUE, print_table = TRUE)

# TPR
df2boxplot(file="model_tpr.csv", ylab="TPR", methods = c("pglmm", "pglmmAIC", "glmnet", "glmnetPC","ggmix"), geo = c("UKBB"), kin_ = kin, change_labels = TRUE, print_table = TRUE)

# Precision
df2boxplot(file="model_ppv.csv", ylab="Precision", methods = c("pglmm", "pglmmAIC", "glmnet", "glmnetPC","ggmix"), geo = c("UKBB"), kin_ = kin, change_labels = TRUE, print_table = TRUE)

#----------------------------
# Supplementary simulations
#----------------------------
supp_df <- read.csv("supp.csv")
df <- rbind(
  cbind(mse=supp_df[, "pglmm_mse_median"], method="pglmm", K=1:100),
  cbind(mse=supp_df[, "pglmm_supp_mse_median"], method="pglmm_hessian", K=1:100),
  cbind(mse=supp_df[, "glmnet_mse_median"], method="glmnetPC", K=1:100)
) %>% as.data.frame()

ggplot(df, aes(y=as.numeric(mse), x=as.numeric(K), colour=method, linetype=method)) +
  geom_line() +
  labs(y=expression(paste("MSE (", hat(beta), ")")), x="lambda index")+
  scale_colour_grey(start=0.7, end=0.1)+
  theme_bw() +
  theme(plot.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank() )+
  #theme(panel.border= element_blank())+
  theme(axis.line.x = element_line(color="black", size = 0.5),
        axis.line.y = element_line(color="black", size = 0.5))
