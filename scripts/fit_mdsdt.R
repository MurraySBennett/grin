{
  # https://github.com/hawkrobe/mdsdt/tree/master
  # https://www.sciencedirect.com/science/article/pii/S0022249616300219#s000100
  rm(list=ls())
  library(tidyverse)
  library(here)
  
  #mdsdt requirements - she's probably in need of a review and update -- also check in with Allan Collins, I think he's used this (a bit, and recently).
  source(here("src", "utils", "mdsdt_grt_base.R"))
  library(mnormt)
  library(ellipse)
  library(polycor)
  
  library(foreach)
  library(doParallel)
  sim_data <- read_csv(here("data", "simulated_grt", "grt_dataset_testsplit.csv"))

  convert_model_names <- function(model_names) {
    mapping <- list(
      "pi_ps_ds" = "PI, PS, DS",
      "pi_psa_ds" = "PI, PS(A), DS",
      "pi_psb_ds" = "PI, PS(B), DS",
      "rho1_ps_ds" = "1_RHO, PS, DS",
      "rho1_psa_ds" = "1_RHO, PS(A), DS",
      "rho1_psb_ds" = "1_RHO, PS(B), DS",
      "pi_ds" = "PI, DS",
      "ps_ds" = "PS, DS",
      "rho1_ds" = "1_RHO, DS",
      "psa_ds" = "PS(A), DS",
      "psb_ds" = "PS(B), DS",
      "ds" = "DS"
    )
    unmatched_names <- setdiff(model_names, names(mapping))
    if (length(unmatched_names) > 0) {
      warning("The following model names were not found in the mapping: ",
              paste(unmatched_names, collapse = ", "))
    }
    converted_names <- model_names
    for (original_name in names(mapping)) {
      converted_names[converted_names == original_name] <- mapping[[original_name]]
    }
    return(converted_names)
  }
  # sim_data$model_name <- convert_model_names(sim_data$model_name)
  
  
}

{
  create_cm_list <- function(df, set_prop = TRUE) {
    cm_columns <- names(df)[grepl("^cm_s[1-4]_r[1-4]$", names(df))]
    if (length(cm_columns) == 0) {
      stop("No confusion matrix columns found in the data frame.")
    }
    cm_data <- df[, cm_columns]
    cm_list <- list()
    for (i in 1:nrow(cm_data)) {
      # Extract the vector of flattened confusion matrix values for the current row
      cm_vector <- as.numeric(cm_data[i, ])
      current_cm <- matrix(cm_vector, nrow = 4, ncol = 4, byrow = TRUE)
      # Assign row and column names for clarity
      colnames(current_cm) <- c("resp1", "resp2", "resp3", "resp4")
      rownames(current_cm) <- c("stim1", "stim2", "stim3", "stim4")
      
      # mdsdt ordering is: a1b1, a1b2, a2b1, a2b2 cf what I generated (a1b1, a2b1, a1b2, a2b2)
      current_cm[2:3, ] <- current_cm[3:2, ]
      current_cm[, 2:3] <- current_cm[, 3:2]
      cm_list[[i]] <- current_cm
      if (set_prop) {
        cm_list[[i]] <- round(current_cm / rowSums(current_cm), 3)
      }
    }
    return(cm_list)
  }
  cmats <- create_cm_list(sim_data, set_prop = FALSE) # mdsdt uses counts 
 
  get_mets <- function(fit, name) {
    tibble(
      model = name,
      AIC = fit$AIC,
      AIC.c = fit$AIC.c,
      BIC = fit$BIC,
      model_object = list(fit)
    )
  }
  
  fit_models <- function(cm) {
    fit_metrics <- tibble(model = character(), AIC = numeric(), AIC.c = numeric(), BIC = numeric(), model_object = list())
    
    fit.pi_ps_ds <- fit.grt(cm, PI = 'all', PS_x = T, PS_y = T)
    fit_metrics <- bind_rows(fit_metrics, get_mets(fit.pi_ps_ds, 'pi_ps_ds'))
    fit.pi_psa_ds <- fit.grt(cm, PI = 'all', PS_x = T, PS_y = F)
    fit_metrics <- bind_rows(fit_metrics, get_mets(fit.pi_psa_ds, 'pi_psa_ds'))
    fit.pi_psb_ds <- fit.grt(cm, PI = 'all', PS_x = F, PS_y = T)
    fit_metrics <- bind_rows(fit_metrics, get_mets(fit.pi_psb_ds, 'pi_psb_ds'))
    
    fit.rho1_ps_ds <- fit.grt(cm, PI = 'same_rho', PS_x = T, PS_y = T)
    fit_metrics <- bind_rows(fit_metrics, get_mets(fit.rho1_ps_ds, 'rho1_ps_ds'))
    fit.rho1_psa_ds <- fit.grt(cm, PI = 'same_rho', PS_x = T, PS_y = F)
    fit_metrics <- bind_rows(fit_metrics, get_mets(fit.rho1_psa_ds, 'rho1_psa_ds'))
    fit.rho1_psb_ds <- fit.grt(cm, PI = 'same_rho', PS_x = F, PS_y = T)
    fit_metrics <- bind_rows(fit_metrics, get_mets(fit.rho1_psb_ds, 'rho1_psb_ds'))
    
    fit.ps_ds <- fit.grt(cm, PI = 'none', PS_x = T, PS_y = T)
    fit_metrics <- bind_rows(fit_metrics, get_mets(fit.ps_ds, 'ps_ds'))
    fit.psa_ds <- fit.grt(cm, PI = 'none', PS_x = T, PS_y = F)
    fit_metrics <- bind_rows(fit_metrics, get_mets(fit.psa_ds, 'psa_ds'))
    fit.psb_ds <- fit.grt(cm, PI = 'none', PS_x = F, PS_y = T)
    fit_metrics <- bind_rows(fit_metrics, get_mets(fit.psb_ds, 'psb_ds'))
    
    fit.pi_ds <- fit.grt(cm, PI = 'all', PS_x = F, PS_y = F)
    fit_metrics <- bind_rows(fit_metrics, get_mets(fit.pi_ds, 'pi_ds'))
    fit.rho1_ds <- fit.grt(cm, PI = 'same_rho', PS_x = F, PS_y = F)
    fit_metrics <- bind_rows(fit_metrics, get_mets(fit.rho1_ds, 'rho1_ds'))
    fit.ds <- fit.grt(cm, PI = 'none', PS_x = F, PS_y = F)
    fit_metrics <- bind_rows(fit_metrics, get_mets(fit.ds, 'ds'))
    
    # Arrange by BIC, AIC.c, and AIC to find the best-fitting model
    fit_metrics <- fit_metrics %>% arrange(BIC, AIC.c, AIC)
    
    # Return both the metrics and the best-fitting model object
    return(list(
      table = fit_metrics,
      best_model = fit_metrics$model_object[[1]]
    ))
  }
}

{
  fit_result <- fit_models(cm)
  plot(fit_result$best_model)
}
