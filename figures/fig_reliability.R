suppressMessages({library(dplyr); library(readr)})
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), "theme.R"))
d <- read_csv("analysis/reliability_bins.csv", show_col_types = FALSE) %>%
  filter(n >= 10) %>%
  mutate(class = recode(model, "jev" = "Jev",
                        "google_gemini-3.8-flash" = "Gemini 3.8 Flash",
                        "local_rlcd-0.6b" = "Qwen3-0.6B-RLCD"))

p <- ggplot(d, aes(mean_conf, acc, colour = class)) +
  geom_abline(colour = "grey70", linewidth = 0.3) +
  geom_line(linewidth = 0.35, alpha = 0.8) +
  geom_point(aes(size = n), alpha = 0.7, stroke = 0) +
  scale_size_area(max_size = 2.2) +
  scale_colour_manual(values = c("Jev" = COL_JEV,
                                 "Gemini 3.8 Flash" = COL_LLM,
                                 "Qwen3-0.6B-RLCD" = COL_LOCAL)) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  facet_wrap(~task, ncol = 6) +
  labs(x = "Stated confidence (bin mean)", y = "Empirical accuracy") +
  theme_dm(base_size = 8) +
  theme(legend.position = "bottom", legend.title = element_blank(),
        axis.text = element_text(size = 5.5))

save_fig(p, "fig_reliability", w = 7.0, h = 4.6)
