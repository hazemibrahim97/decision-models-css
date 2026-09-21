suppressMessages({library(dplyr); library(readr); library(tidyr); library(scales)})
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), "theme.R"))

conf_tasks <- read_csv("analysis/cell_metrics.csv", show_col_types = FALSE) %>%
  filter(split == "confirmatory") %>% distinct(task) %>% pull(task)

d <- read_csv("analysis/cascade.csv", show_col_types = FALSE) %>%
  filter(task %in% conf_tasks) %>%
  mutate(L = recode(llm,
                    "google_gemini-3.8-flash" = "Cascade to Gemini 3.8 Flash",
                    "google_gemma-4-31b-it" = "Cascade to Gemma 4 31B",
                    "anthropic_claude-haiku-4.5" = "Cascade to Haiku 4.5"),
         cost1k_cas = cost_cascade / n * 1000,
         cost1k_llm = cost_llm_alone / n * 1000,
         cost1k_jev = cost_jev_alone / n * 1000)

med <- d %>% group_by(L, t) %>%
  summarise(Accuracy = median(acc_cascade),
            `Cost per 1,000 items (USD)` = median(cost1k_cas), .groups = "drop") %>%
  pivot_longer(c(Accuracy, `Cost per 1,000 items (USD)`),
               names_to = "metric", values_to = "y")

refs <- d %>% group_by(L) %>%
  summarise(acc_llm = median(acc_llm_alone), acc_jev = median(acc_jev_alone),
            cost_llm = median(cost1k_llm), cost_jev = median(cost1k_jev),
            .groups = "drop") %>%
  pivot_longer(-L) %>%
  mutate(metric = ifelse(startsWith(name, "acc"), "Accuracy",
                         "Cost per 1,000 items (USD)"),
         who = ifelse(endsWith(name, "llm"), "LLM alone", "Jev alone"))
library(patchwork)
COL_COST <- "#0072B2"

wide <- med %>% pivot_wider(names_from = metric, values_from = y) %>%
  rename(acc = Accuracy, cost = `Cost per 1,000 items (USD)`)
refw <- refs %>% select(L, name, value) %>% pivot_wider(names_from = name)

panel <- function(Lval) {
  m <- filter(wide, L == Lval)
  r <- filter(refw, L == Lval)
  alo <- min(m$acc, r$acc_jev, r$acc_llm) - 0.005
  ahi <- max(m$acc, r$acc_jev, r$acc_llm) + 0.005
  cmax <- max(m$cost, r$cost_llm) * 1.05
  sc <- function(cost) alo + cost / cmax * (ahi - alo)
  ggplot(m, aes(t)) +
    geom_hline(aes(yintercept = sc(r$cost_llm), colour = "LLM alone (cost)"),
               linetype = "dotted", linewidth = 0.35, alpha = 0.7) +
    geom_hline(aes(yintercept = sc(r$cost_jev), colour = "Jev alone (cost)"),
               linetype = "dotted", linewidth = 0.35, alpha = 0.7) +
    geom_hline(aes(yintercept = r$acc_llm, colour = "LLM alone (accuracy)"),
               linetype = "dashed", linewidth = 0.35, alpha = 0.8) +
    geom_hline(aes(yintercept = r$acc_jev, colour = "Jev alone (accuracy)"),
               linetype = "dashed", linewidth = 0.35, alpha = 0.8) +
    geom_line(aes(y = acc), colour = COL_REF, linewidth = 0.45) +
    geom_point(aes(y = acc), colour = COL_REF, size = 1.6) +
    geom_line(aes(y = sc(cost)), colour = COL_COST, linewidth = 0.45) +
    geom_point(aes(y = sc(cost)), colour = COL_COST, size = 1.6, shape = 15) +
    scale_colour_manual(
      values = c("LLM alone (accuracy)" = COL_LLM,
                 "Jev alone (accuracy)" = COL_JEV,
                 "LLM alone (cost)" = COL_LLM,
                 "Jev alone (cost)" = COL_JEV),
      name = NULL,
      breaks = c("Jev alone (accuracy)", "LLM alone (accuracy)",
                 "Jev alone (cost)", "LLM alone (cost)"),
      guide = guide_legend(override.aes = list(
        linetype = c("dashed", "dashed", "dotted", "dotted")))) +
    scale_x_continuous(breaks = c(0.5, 0.6, 0.7, 0.8, 0.9, 0.95)) +
    scale_y_continuous(
      name = "Accuracy (circles)",
      limits = c(alo, ahi),
      sec.axis = sec_axis(~ (. - alo) / (ahi - alo) * cmax,
                          name = "Cost per 1,000 items, USD (squares)")) +
    labs(x = "Jev confidence threshold t", title = Lval) +
    theme_dm() +
    theme(legend.position = "bottom",
          plot.title = element_text(size = 8, face = "bold", hjust = 0.5),
          axis.title.y.right = element_text(colour = COL_COST),
          axis.text.y.right = element_text(colour = COL_COST))
}

LS <- c("Cascade to Gemini 3.8 Flash", "Cascade to Gemma 4 31B",
        "Cascade to Haiku 4.5")
fig <- panel(LS[1]) + panel(LS[2]) + panel(LS[3]) +
  plot_layout(guides = "collect") & theme(legend.position = "bottom")

save_fig(fig, "fig_cascade", w = 9.0, h = 3.3)
