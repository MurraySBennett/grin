{
  rm(list=ls())
  library(tidyverse)
  library(here)
  library(grtools)
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
  sim_data$model_name <- convert_model_names(sim_data$model_name)
  
  
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
      cm_list[[i]] <- current_cm
      if (set_prop) {
        cm_list[[i]] <- round(current_cm / rowSums(current_cm), 3)
      }
    }
    return(cm_list)
  }
  
  fit_cm <- function(cmat, idx) {
    start_time <- Sys.time()
    fit_result <- grt_hm_fit(cmat)
    end_time <- Sys.time()
    fit_time_sec <- as.numeric(difftime(end_time, start_time, units = "secs"))
    
    best_model <- fit_result$best_model
    best_model_name <- gsub("GRT-\\{|\\}", "", best_model$model)
    
    means <- as.numeric(t(best_model$means))
    # x_means <- matrix(best_model$means[,1],nrow=1)
    # y_means <- matrix(best_model$means[,2],nrow=1)
    covmats <- unlist(best_model$covmat)
    decision_bounds <- c(best_model$a1, best_model$a2)
    
    best_fit_params <- c(means, covmats, decision_bounds)
    
    return(list(
      sample_id = idx,
      predicted_model = best_model_name,
      fit_time_seconds = fit_time_sec,
      best_fit_params = best_fit_params,
      convergence = best_model$convergence,
      message = best_model$message
    ))
  }

  cmats <- create_cm_list(sim_data)
}

{
  num_cores <- detectCores() - 1 
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)
  
  message("Fitting ", length(cmats), " matrices using ", num_cores, " cores...")
  
  fit_results <- foreach(
    i = 1:10, # Use length(cmats) for full run
    .packages = c("grtools"),
    .inorder = TRUE#,
    # .export = c("fit_cm", "cmats", "sim_data") 
  ) %dopar% {
    fit_cm(cmats[[i]], sim_data$sample_id[i])
  }
  stopCluster(cl)
  message("Fitting complete.")

  all_fit_results <- tibble(
    sample_id = sapply(fit_results, function(x) x$sample_id),
    predicted_model = sapply(fit_results, function(x) x$predicted_model),
    fit_time_seconds = sapply(fit_results, function(x) x$fit_time_seconds),
    convergence = sapply(fit_results, function(x) x$convergence),
    message = sapply(fit_results, function(x) x$message),
    best_fit_params = sapply(fit_results, function(x) list(x$best_fit_params))
  )
  
  # print(all_fit_results)
  # View(all_fit_results)
}
