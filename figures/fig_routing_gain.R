suppressMessages({library(dplyr); library(readr); library(tidyr); library(ggrepel)})
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), "theme.R"))

d <- read_csv("analysis/cell_metrics.csv", show_col_types = FALSE) %>%
  filter(model == "jev", !is.na(`acc0.9`)) %>%
  mutate(gain = `acc0.9` - acc) %>%
  select(task, split, gain, ECE = ece, `Base accuracy` = acc) %>%
  pivot_longer(c(ECE, `Base accuracy`), names_to = "xvar", values_to = "x")

p <- ggplot(d, aes(x, gain)) +
  geom_hline(yintercept = 0, colour = "grey70", linewidth = 0.3) +
  geom_point(aes(shape = split), colour = COL_JEV, size = 1.8) +
  geom_text_repel(aes(label = task), size = 2.0, family = "Helvetica",
                  colour = "grey40", seed = 20260920, max.overlaps = 20) +
  scale_shape_manual(values = c(confirmatory = 16, discovery = 1)) +
  facet_wrap(~xvar, scales = "free_x") +
  labs(x = NULL, y = "Routing gain at t = 0.9 (acc@0.9 - base acc)") +
  theme_dm() +
  theme(legend.position = "bottom", legend.title = element_blank())

save_fig(p, "fig_routing_gain", w = 6.2, h = 3.2)
