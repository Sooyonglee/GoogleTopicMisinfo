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
  rename('predict_label_conditional1' = 'predict_label')

prompt <- 'prompt2'
gpt_conditional_prompt2 <- fread(paste0(project_dir, 'data/', gpt, '/groupharm-conditional-results-', prompt, '.csv')) %>%
  rename('predict_label_conditional1' = 'predict_label')


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

write.csv(first_table, file = '~/Desktop/gender_counts.csv', row.names = F)

##############################
### --- Data Cleaning
##############################

### Creates topic level data frame with relevant measures

gpt_rq1 <- gpt_conditional %>% # join data sources
  left_join(gpt_base) %>%
  left_join(raw_data, by = c('claim'='source')) %>%
  mutate(conditional_squared_error = (true_label - predict_label_conditional)^2) %>%
  group_by(topic, gender) %>%
  # capture relevant variables
  summarise(conditional_MSE = mean(conditional_squared_error),
            mean_true_label = mean(true_label),
            mean_conditional_label = mean(predict_label_conditional),
            sd_true_label = sd(true_label),
            sd_conditional_label = sd(predict_label_conditional)) %>%
  # change data shape to allow for Female/Male comparisons
  pivot_wider(., 
              id_cols = topic, 
              names_from = gender, 
              values_from = c(conditional_MSE, 
                              mean_true_label, 
                              mean_conditional_label, 
                              sd_true_label, 
                              sd_conditional_label)) %>%
  # Create Variables for Plotting
  mutate(diff_between_gender_means_conditional = mean_conditional_label_Female - mean_conditional_label_Male,
         diff_between_gender_means_true = mean_true_label_Female - mean_true_label_Male,
         diff_within_gender_male = mean_conditional_label_Male - mean_true_label_Male,
         diff_within_gender_female = mean_conditional_label_Female - mean_true_label_Female)


#######################
############### Plots
#######################

###############
####### Exp 1a: For hypothesized topics, examine accuracy of conditioned prompts compared to human data.
###############

plot_data <- gpt_rq1 %>%
  filter(topic %in% hyp_topics) %>%
  select(topic, mean_conditional_label_Female, mean_conditional_label_Male, mean_true_label_Female, mean_true_label_Male) %>%
  melt(., id.vars = 'topic') %>%
  mutate(data_type = ifelse(grepl('true', variable), 'Human Data', 'Cond. Prompt'),
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
  select(topic, mean_conditional_label_Female, mean_conditional_label_Male, mean_true_label_Female, mean_true_label_Male) %>%
  melt(., id.vars = 'topic') %>%
  mutate(data_type = ifelse(grepl('true', variable), 'Human Data', 'Cond. Prompt'),
         gender = ifelse(grepl('Female', variable), 'Female', 'Male'))

g2 <- ggplot(plot_data, aes(x = data_type, y = value, fill = gender)) +
  geom_line(aes(group = interaction(data_type, topic)), linewidth = 1, alpha = 0.8) +
  geom_point(shape = 21, size = 4) +
  facet_wrap(~topic) +
  coord_flip() +
  ylab('Mean Rating for Perceived Group Harm\n(1-6 scale)') +
  xlab('')

ggsave(g2, filename = paste0(project_dir, 'output/means_plot_1b.png'), width = 12, height = 6)


###############
######### Exp 2: Base prompts 
###############

plot_data <- gpt_rq2 %>%
  filter(topic %in% hyp_topics)

g3 <- ggplot(plot_data) + 
  geom_bar(aes(x = gender, y = mean_squared_error, fill = gender), stat = 'identity') +
  facet_wrap(~topic) +
  coord_flip() +
  ylab('Mean Squared Error Between Base Prompt Rating and Human Assessment') +
  xlab('') +
  theme(axis.text.y = element_blank())

plot_data <- gpt_sq2
