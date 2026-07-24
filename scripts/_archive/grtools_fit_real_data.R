# get grtools to fit some existing matrices and use those parameter spaces to
# guide the network hyperparemters?

{
  rm(list=ls())
  library(here)
  library(grtools)
}

{
  # grtools ?grt_hm_fit data:
  grtools_cmat <- matrix(c(140, 36, 34, 40,
                   89, 91, 4, 66,
                   85, 5, 90, 70,
                   20, 59, 8, 163),
                 nrow=4, ncol=4, byrow=TRUE)
  
  grtools_windcmats <- list(matrix(c(161,41,66,32,24,147,64,65,37,41,179,43,14,62,22,202),nrow=4,ncol=4,byrow=TRUE))
  grtools_windcmats[[2]] <- matrix(c(126,82,67,25,8,188,54,50,34,75,172,19,7,103,14,176),nrow=4,ncol=4,byrow=TRUE)
  grtools_windcmats[[3]] <- matrix(c(117,64,89,30,11,186,69,34,21,81,176,22,10,98,30,162),nrow=4,ncol=4,byrow=TRUE)
  grtools_windcmats[[4]] <- matrix(c(168,57,47,28,15,203,33,49,58,54,156,32,9,96,9,186),nrow=4,ncol=4,byrow=TRUE)
  grtools_windcmats[[5]] <- matrix(c(169,53,53,25,34,168,69,29,38,48,180,34,19,44,60,177),nrow=4,ncol=4,byrow=TRUE)
  
  # Perform model fit and selection
  grtools_fit <- grt_hm_fit(grtools_cmat)
  grtools_fits<- lapply(grtools_windcmats, grt_hm_fit)
  
  
  # variances of 1 for a 1rho model. Means are equal for ps conditions, not varied.
  
}

{
  mdsdt_to_grtools <- function(path){
    data_object <- load(path)
    cm <- get(data_object)
    cm[2:3,] = cm[3:2,]
    cm[,2:3] = cm[,2:3]
    return(cm)
  }
  mdsdt_files <- list.files(here("data", "mdsdt_data"), pattern="*.rda", full.names = T)
  mdsdt_cms <- lapply(mdsdt_files, mdsdt_to_grtools)
  
  mdsdt_fits <- lapply(mdsdt_cms, grt_hm_fit)
}


{
  ## My trashy melanoma data
  ab_data <- readRDS(here("..", "melanoma-identification", "grt", "model_outputs", "ab_models.rds"))
  ac_data <- readRDS(here("..", "melanoma-identification", "grt", "model_outputs", "ac_models.rds"))
  bc_data <- readRDS(here("..", "melanoma-identification", "grt", "model_outputs", "bc_models.rds"))
  
}
