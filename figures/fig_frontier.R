suppressMessages({library(dplyr); library(readr); library(ggrepel); library(scales)
                  library(patchwork)})
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), "theme.R"))

d <- read_csv("analysis/frontier.csv", show_col_types = FALSE) %>%
  filter(kind != "local") %>%
  mutate(
    cost_plot = cost_per_1k,
    class = ifelse(model == "jev", "jev", "llm"),
    label = recode(model,
      "jev" = "Jev",
      "local_rlcd-0.6b" = "Qwen3-0.6B-RLCD",
      "local_qwen3-base" = "Qwen3-0.6B base",
      "anthropic_claude-haiku-4.5" = "Haiku 4.5",
      "anthropic_claude-sonnet-5" = "Sonnet 5",
      "anthropic_claude-opus-5" = "Opus 5",
      "anthropic_claude-fable-5.1" = "Fable 5.1",
      "google_gemini-3.5-flash-lite" = "Gemini Flash-Lite",
      "google_gemini-3.8-flash" = "Gemini 3.8 Flash",
      "google_gemini-3.1-pro-preview" = "Gemini 3.1 Pro",
      "google_gemma-4-31b-it" = "Gemma 4 31B",
      "meta-llama_llama-4-scout" = "Llama 4 Scout",
      "meta-llama_llama-4-maverick" = "Llama 4 Maverick",
      "mistralai_mistral-medium-3-5" = "Mistral Medium",
      "moonshotai_kimi-k2.6" = "Kimi K2.6",
      "openai_gpt-5.6-luna" = "GPT-5.6 Luna",
      "openai_gpt-5.6-sol" = "GPT-5.6 Sol",
      "openai_gpt-5.6-terra" = "GPT-5.6 Terra",
      "openai_gpt-oss-120b" = "GPT-OSS 120B",
      "deepseek_deepseek-v3.2" = "DeepSeek V3.2",
      "qwen_qwen3-235b-a22b-2507" = "Qwen3 235B",
      "z-ai_glm-5.3" = "GLM 5.3"))

front <- d %>% arrange(cost_plot, desc(median_f1)) %>%
  filter(median_f1 == cummax(median_f1))

p <- ggplot(d, aes(cost_plot, median_f1)) +
  geom_step(data = front, direction = "hv", colour = "grey75", linewidth = 0.4) +
  geom_point(aes(colour = class, shape = class), size = 2.2) +
  geom_text_repel(aes(label = label, colour = class), size = 2.4,
                  family = "Helvetica", seed = 20260920, max.overlaps = 20,
                  box.padding = 0.3, min.segment.length = 0.15,
                  nudge_y = ifelse(d$model == "jev", 0.03, 0),
                  nudge_x = ifelse(d$model == "jev", -0.25, 0),
                  segment.colour = "grey70", show.legend = FALSE) +
  scale_colour_manual(values = c(jev = COL_JEV, local = COL_LOCAL, llm = COL_LLM)) +
  scale_shape_manual(values = c(jev = 17, local = 17, llm = 16)) +
  scale_x_log10(labels = label_dollar(accuracy = 0.01),
                breaks = c(0.01, 0.1, 1, 10), limits = c(0.012, 20)) +
  labs(x = "Measured cost per 1,000 items (USD, log scale)",
       y = "Median macro-F1 (15 evaluation tasks)") +
  theme_dm()

cm <- read_csv("analysis/cell_metrics.csv", show_col_types = FALSE) %>%
  filter(split == "confirmatory", kind != "local") %>%
  mutate(eff_task = (macro_f1 * 100) / (cost / n * 1000 * 100))

eff <- cm %>% group_by(model) %>%
  summarise(mid = median(eff_task),
            lo = quantile(eff_task, 0.25),
            hi = quantile(eff_task, 0.75), .groups = "drop") %>%
  left_join(d %>% select(model, label, class), by = "model") %>%
  mutate(name = factor(label, levels = label[order(mid)]))

pb <- ggplot(eff, aes(mid, name, colour = class)) +
  geom_vline(xintercept = 1, colour = COL_REF, linewidth = 0.35,
             linetype = "dashed", alpha = 0.35) +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.25, linewidth = 0.35) +
  geom_point(size = 1.9) +
  geom_text(aes(x = hi, label = ifelse(mid >= 1, sprintf("%.1f", mid),
                                       sprintf("%.2f", mid))),
            hjust = -0.3, size = 2.1, family = "Helvetica", colour = "grey30") +
  scale_colour_manual(values = c(jev = COL_JEV, llm = COL_LLM)) +
  scale_x_log10(labels = comma, expand = expansion(mult = c(0.06, 0.12))) +
  labs(x = "Macro-F1 points per cent per 1,000 items (log)", y = NULL) +
  theme_dm() +
  theme(axis.text.y = element_text(
    colour = ifelse(arrange(eff, mid)$class == "jev", COL_JEV, "grey20")))

fig <- p + pb + plot_layout(widths = c(1.15, 1)) +
  plot_annotation(tag_levels = "A") &
  theme(plot.tag = element_text(size = 10, face = "bold", family = "Helvetica"))

save_fig(fig, "fig_frontier", w = 8.6, h = 3.6)
