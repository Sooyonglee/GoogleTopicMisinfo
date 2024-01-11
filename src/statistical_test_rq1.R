library(dplyr)
library(data.table)
library(ggplot2)
library(tidyr)


##############################
### --- Bootstrap Function - claim-level resampling with replacement
##############################

bootstrap_function <- function(gpt_datatable, raw_datatable, hyp_level, iters) {
  
  
  gpt_joined <- gpt_datatable[raw_datatable, on = .(claim = source)]
  # Bootstrapping parameters
  nrow_boot <- nrow(gpt_joined)
  # Pre-allocate list to store results
  bootstrap_results_list <- vector("list", iters)
  # Bootstrapping loop
  for (i in 1:iters) {
    
    # create data.table of resampled claims
    resampled_claims <- raw_datatable[, .(source)][, .(claim = sample(source, .N, replace = TRUE))]
    
    # join annotation data to resampled claims
    resampled_dt <- gpt_joined[resampled_claims, on = 'claim', allow.cartesian = TRUE]
    
    # reshape data to calculate gender-level differences
    gpt_groupharm_topic <- resampled_dt[, .(mean_true = mean(true_label),
                                            mean_gpt = mean(predict_label_conditional)), by = .(claim,topic, gender)] %>%
      pivot_wider(., 
                  id_cols = c(topic, claim), 
                  names_from = gender, 
                  values_from = c(mean_true, mean_gpt)) %>%
      mutate(test_statistic = abs(mean_gpt_Female - mean_gpt_Male) - abs(mean_true_Female - mean_true_Male)) %>%
      group_by(topic) %>%
      summarise(mean_true_Female = mean(mean_true_Female),
                mean_true_Male = mean(mean_true_Male),
                mean_gpt_Female = mean(mean_gpt_Female),
                mean_gpt_Male = mean(mean_gpt_Male),
                test_statistic = mean(test_statistic)) %>%
      setDT()
    
    gpt_groupharm_topic[, iter := i]
    setcolorder(gpt_groupharm_topic, c('topic',
                                       'mean_true_Female', 
                                       'mean_true_Male', 
                                       'mean_gpt_Female', 
                                       'mean_gpt_Male', 
                                       'test_statistic',
                                       'iter'))
    
    # Store results
    bootstrap_results_list[[i]] <- gpt_groupharm_topic
    
    if (i %% 100 == 0) {
      cat("Iteration:", i, "\n")
    }
  }
  
  bootstrap_results_raw <- rbindlist(bootstrap_results_list)
  
  bootstrap_results_clean <- bootstrap_results_raw %>%
    group_by(topic) %>%
    summarise(p_value = 1 - round(mean(ifelse(test_statistic > hyp_level, 1, 0)), 9),
              mean_test_statistic = mean(test_statistic),
              se_test_statistic = sd(test_statistic))
  
  return(list(clean_results = bootstrap_results_clean, raw_results = bootstrap_results_raw))
}


run_bootstrap <- function(project_dir, hyp_topics, hyp_level, gpt, prompt, iters) {
  gpt_base <- fread(paste0(project_dir, 'data/', gpt, '/groupharm-base-results-', prompt, '.csv'))[, .(predict_label = mean(predict_label)), by = .(claim)]
  
  gpt_conditional <- fread(paste0(project_dir, 'data/', gpt, '/groupharm-conditional-results-', prompt, '.csv')) %>%
    rename('predict_label_conditional' = 'predict_label')
  
  raw_data <- fread(paste0(project_dir, 'data/GroundTruthPreExperiment.csv')) %>%
    mutate(topic = ifelse(is.na(topic), 'Abortion', topic),
           topic = ifelse(grepl('Gold', topic), 'Gold', topic),
           claimID = seq.int(nrow(.))) %>%
    setDT()
  
  bootstrap_data <- bootstrap_function(gpt_datatable = gpt_conditional, 
                                       raw_datatable = raw_data, 
                                       iters = iters, 
                                       hyp_level = hyp_level)
  
  raw_results <- bootstrap_data$raw_results
  clean_results <- bootstrap_data$clean_results
  return(list(raw_results = raw_results, clean_results = clean_results))
}

##############################
### --- Main
##############################
project_dir <- '~/Desktop/GoogleTopicMisinfo/'
hyp_topics <- c('Abortion', 'Illegal Immigration', 'Black Americans', 'LGBTQ')
gpt <- 'gpt-35'
iters <- 10000
hyp_level <- 0


# - run for prompt 1

prompt <- 'prompt1'

bootstrap_data <- run_bootstrap(project_dir = project_dir,
                                hyp_topics = hyp_topics,
                                gpt = gpt,
                                prompt = prompt,
                                iters = iters,
                                hyp_level = hyp_level)

write.csv(bootstrap_data$clean_results, file = paste0(project_dir,'output/rq1_', prompt, '_stats.csv'), row.names = F)

# - run for prompt 2

prompt <- 'prompt2'

bootstrap_data <- run_bootstrap(project_dir = project_dir,
                                hyp_topics = hyp_topics,
                                gpt = gpt,
                                prompt = prompt,
                                iters = iters,
                                hyp_level = hyp_level)

write.csv(bootstrap_data$clean_results, file = paste0(project_dir,'output/rq1_', prompt, '_stats.csv'), row.names = F)