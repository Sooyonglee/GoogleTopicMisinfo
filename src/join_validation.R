k <- (resampled_claims %>% 
  group_by(claim) %>% 
  summarise(count = n())) %>% 
  left_join(gpt_conditional %>% 
              group_by(claim) %>% 
              summarise(annotations = n())) %>% 
  mutate(expected_number = annotations*count) %>%
  left_join((resampled_dt %>% 
              group_by(claim) %>% 
              summarise(actual_count = n())))
