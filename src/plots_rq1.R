library(dplyr)
library(data.table)
library(ggplot2)
library(tidyr)

project_dir <- '~/Desktop/GoogleTopicMisinfo/'
hyp_topics <- c('Abortion', 'Illegal Immigration', 'Black Americans', 'LGBTQ')
gpt <- 'gpt-35'

# Load data
prompt <- 'prompt1'
gpt_conditional_prompt1 <- fread(paste0(project_dir, 'data/', gpt, '/groupharm-conditional-results-', prompt, '.csv')) %>%
  rename('predict_label_conditional1' = 'predict_label') %>%
  select(claim, gender, true_label, predict_label_conditional1) %>%
  group_by(claim, gender) %>%
  summarise(mean_conditional_label1 = mean(predict_label_conditional1),
            mean_true_label = mean(true_label)) %>%
  ungroup()

prompt <- 'prompt2'
gpt_conditional_prompt2 <- fread(paste0(project_dir, 'data/', gpt, '/groupharm-conditional-results-', prompt, '.csv')) %>%
  rename('predict_label_conditional2' = 'predict_label') %>%
  select(claim, gender, predict_label_conditional2) %>%
  group_by(claim, gender) %>%
  summarise(mean_conditional_label2 = mean(predict_label_conditional2)) %>%
  ungroup()


raw_data <- fread(paste0(project_dir, 'data/GroundTruthPreExperiment.csv')) %>%
  mutate(topic = ifelse(is.na(topic), 'Abortion', topic),
         topic = ifelse(grepl('Gold', topic), 'Gold', topic),
         claimID = seq.int(nrow(.))) %>%
  setDT()

# Counts by gender
first_table <- raw_data %>%
  left_join(gpt_conditional_prompt1, by = c('source'='claim')) %>%
  group_by(gender, topic) %>%
  summarise(count = n()) %>%
  dcast(., topic ~ gender)

write.csv(first_table, file = paste0(project_dir, 'output/gender_counts.csv'), row.names = F)

##############################
### --- Data Cleaning
##############################

### Creates topic level data frame with relevant measures

gpt_rq1 <- gpt_conditional_prompt1 %>% # join data sources
  left_join(gpt_conditional_prompt2) %>%
  left_join(raw_data, by = c('claim'='source')) %>%
  group_by(topic, gender) %>%
  summarise(mean_true_label = mean(mean_true_label),
            mean_conditional_label1 = mean(mean_conditional_label1),
            mean_conditional_label2 = mean(mean_conditional_label2)) %>%
  # change data shape to allow for Female/Male comparisons
  pivot_wider(., 
              id_cols = topic, 
              names_from = gender, 
              values_from = c(mean_true_label, 
                              mean_conditional_label1,
                              mean_conditional_label2)) 


#######################
############### Plots
#######################

###############
####### Exp 1a: For hypothesized topics, examine accuracy of conditioned prompts compared to human data.
###############

plot_data <- gpt_rq1 %>%
  filter(topic %in% hyp_topics) %>%
  select(topic, mean_conditional_label1_Female, mean_conditional_label1_Male, mean_conditional_label2_Female, mean_conditional_label2_Male, mean_true_label_Female, mean_true_label_Male) %>%
  melt(., id.vars = 'topic') %>%
  mutate(data_type = ifelse(grepl('true', variable), 'Human Data', 
                            ifelse(grepl('label1', variable), 'Cond. Prompt 1', 'Cond. Prompt 2')),
         data_type = factor(data_type, levels = c('Human Data', 'Cond. Prompt 1', 'Cond. Prompt 2')),
         gender = ifelse(grepl('Female', variable), 'Female', 'Male'))

g1 <- ggplot(plot_data, aes(x = data_type, y = value, fill = gender)) +
  geom_line(aes(group = interaction(data_type, topic)), linewidth = 1, alpha = 0.8) +
  geom_point(shape = 21, size = 4) +
  facet_wrap(~topic) +
  coord_flip() +
  ylab('Mean Rating for Perceived Group Harm\n(1-6 scale)') +
  xlab('')

ggsave(g1, filename = paste0(project_dir, 'output/means_plot_1a.png'), width = 8, height = 6)

###############
####### Exp 1b: For unhypothesized topics, examine accuracy of conditioned prompts compared to human data.
###############

plot_data <- gpt_rq1 %>%
  filter(!(topic %in% hyp_topics)) %>%
  select(topic, mean_conditional_label1_Female, mean_conditional_label1_Male, mean_conditional_label2_Female, mean_conditional_label2_Male, mean_true_label_Female, mean_true_label_Male) %>%
  melt(., id.vars = 'topic') %>%
  mutate(data_type = ifelse(grepl('true', variable), 'Human Data', 
                            ifelse(grepl('label1', variable), 'Cond. Prompt 1', 'Cond. Prompt 2')),
         data_type = factor(data_type, levels = c('Human Data', 'Cond. Prompt 1', 'Cond. Prompt 2')),
         gender = ifelse(grepl('Female', variable), 'Female', 'Male'))

g2 <- ggplot(plot_data, aes(x = data_type, y = value, fill = gender)) +
  geom_line(aes(group = interaction(data_type, topic)), linewidth = 1, alpha = 0.8) +
  geom_point(shape = 21, size = 4) +
  facet_wrap(~topic) +
  coord_flip() +
  ylab('Mean Rating for Perceived Group Harm\n(1-6 scale)') +
  xlab('')

ggsave(g2, filename = paste0(project_dir, 'output/means_plot_1b.png'), width = 12, height = 6)


