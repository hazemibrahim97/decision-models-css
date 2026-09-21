suppressMessages({library(dplyr); library(readr)})
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), "theme.R"))

d <- read_csv("analysis/routing_curves.csv", show_col_types = FALSE) %>%
  mutate(class = recode(model, "jev" = "Jev",
                        "google_gemini-3.8-flash" = "Gemini 3.8 Flash",
                        "local_rlcd-0.6b" = "Qwen3-0.6B-RLCD"))
base <- d %>% group_by(task, class) %>%
  filter(coverage == max(coverage)) %>% ungroup()

p <- ggplot(d, aes(coverage, accuracy, colour = class)) +
  geom_hline(data = base, aes(yintercept = accuracy, colour = class),
             linetype = "dashed", linewidth = 0.25, alpha = 0.6) +
  geom_line(linewidth = 0.4) +
  scale_colour_manual(values = c("Jev" = COL_JEV,
                                 "Gemini 3.8 Flash" = COL_LLM,
                                 "Qwen3-0.6B-RLCD" = COL_LOCAL)) +
  coord_cartesian(ylim = c(0, 1)) +
  facet_wrap(~task, ncol = 6) +
  labs(x = "Coverage (most-confident items first)", y = "Accuracy on covered items") +
  theme_dm(base_size = 8) +
  theme(legend.position = "bottom", legend.title = element_blank(),
        axis.text = element_text(size = 5.5))

save_fig(p, "fig_routing", w = 7.0, h = 4.6)
