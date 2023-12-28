library(dplyr)
library(data.table)
library(ggplot2)
library(tidyr)

project_dir <- '~/Desktop/DiverseButDivisive/'

gpt_base <- fread(paste0(project_dir, 'data/groupharm-base-cleaned-results.csv'))[, .(predict_label = mean(predict_label)), by = .(claim)]
gpt_conditional <- fread(paste0(project_dir, 'data/groupharm-cleaned-results.csv')) %>%
  rename('predict_label_conditional' = 'predict_label')

raw_data <- fread(paste0(project_dir, 'data/GroundTruthPreExperiment.csv')) %>%
  mutate(topic = ifelse(is.na(topic), 'Abortion', topic),
         topic = ifelse(grepl('Gold', topic), 'Gold', topic),
         claimID = seq.int(nrow(.))) %>%
  setDT()


first_table <- raw_data %>%
  left_join(gpt_conditional, by = c('source'='claim')) %>%
  group_by(gender, topic) %>%
  summarise(count = n()) %>%
  dcast(., topic ~ gender)

write.csv(first_table, file = '~/Desktop/gender_counts.csv', row.names = F)
##############################
### --- Bootstrap function
##############################


gpt_joined <- gpt_conditional %>%
  left_join(gpt_base) %>%
  left_join(raw_data, by = c('claim'='source')) %>%
  mutate(base_squared_error = (true_label - predict_label)^2,
         conditional_squared_error = (true_label - predict_label_conditional)^2) %>%
  group_by(topic, gender) %>%
  summarise(base_MSE = mean(base_squared_error),
            conditional_MSE = mean(conditional_squared_error),
            mean_true_label = mean(true_label),
            mean_conditional_label = mean(predict_label_conditional),
            mean_base_label = mean(predict_label),
            sd_true_label = sd(true_label),
            sd_conditional_label = sd(predict_label_conditional),
            sd_base_label = sd(predict_label)) %>%
  pivot_wider(., 
              id_cols = topic, 
              names_from = gender, 
              values_from = c(base_MSE, conditional_MSE, mean_true_label, 
                              mean_conditional_label, mean_base_label, 
                              sd_true_label, sd_conditional_label, sd_base_label)) %>%
  mutate(diff_in_means_conditional = mean_conditional_label_Female - mean_conditional_label_Male,
         diff_in_means_true = mean_true_label_Female - mean_true_label_Male)




#####################
##### RUN BOOTSTRAP
####################

conditional_results <- bootstrap_function(gpt_datatable = gpt_conditional,
                                          raw_datatable = raw_data,
                                          iters = 10000)

base_results <- bootstrap_function(gpt_datatable = gpt_base,
                                   raw_datatable = raw_data,
                                   iters = 10000)


gpt_joined <- gpt_base[raw_data, on = .(claim = source)]
mean_labels_base <- gpt_joined[, .(mean_true = mean(true_label),
                            mean_pred = mean(predict_label)), by = .(topic, gender)]

gpt_joined <- gpt_conditional[raw_data, on = .(claim = source)]
mean_labels_conditional <- gpt_joined[, .(mean_true = mean(true_label),
                                   mean_pred = mean(predict_label)), by = .(topic, gender)]


results <- conditional_results %>%
  left_join(mean_labels_conditional) %>%
  mutate(prompt = 'conditional') %>%
  bind_rows(
    base_results %>%
      left_join(mean_labels_base) %>%
      mutate(prompt = 'base')
  )


#####################
##### Plots
####################

mse_plot <- ggplot(results) + 
  geom_point(aes(x = prompt, y = MSE_boot, fill = prompt), shape=21, size = 4) +
  geom_errorbar(aes(x = prompt, ymin = MSE_boot - SE_boot, ymax = MSE_boot + SE_boot, color = prompt)) + 
  coord_flip() +
  facet_grid(topic ~ gender) + 
  ggtitle("Mean Square Error of ChatGPT and Human Labels\nby Topic and Gender\n(Bootstrapped SE)") +
  theme(axis.text.y = element_blank()) + 
  xlab('') 

ggsave(mse_plot, filename = '~/Desktop/mse_plot.png', height = 14, width = 6)
  
means_data_clean <- means_plotting(results = results)

means_plot <- ggplot(means_data_clean) +
  geom_point(aes(x = variable, y = mean, fill = variable), size = 4, shape = 21) +
  geom_errorbar(aes(x = variable, ymin = mean - se, ymax = mean + se, color = variable)) +
  facet_grid(topic ~ gender) + 
  coord_flip() + 
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        legend.position = 'bottom') + 
  xlab('') +
  ylab('Average Group Harm Rating') +
  ggtitle('Average Group Harm Responses for ChatGPT with Base and Conditioned Prompt\nCompared to Human Responses\nBy Topic and Gender')


ggsave(means_plot, filename = '~/Desktop/means_plot.png', width = 6, height = 14)
