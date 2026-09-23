suppressMessages({library(dplyr); library(readr)})
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), "theme.R"))

SHORT <- c(
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
  "z-ai_glm-5.3" = "GLM 5.3")

main <- read_csv("analysis/cell_metrics.csv", show_col_types = FALSE) %>%
  filter(split == "confirmatory", model %in% names(SHORT)) %>%
  mutate(class = case_when(model == "jev" ~ "jev",
                           kind == "local" ~ "local",
                           TRUE ~ "llm"),
         name = SHORT[model], f1 = macro_f1)

OPEN <- c("local_opendecision" = "NLI-0.4B*", "local_verdict" = "Verdict-0.15B*",
          "local_von" = "Von-0.4B*", "local_laya" = "Laya-0.4B*",
          "local_decider-0.8b" = "decider-0.8b*", "local_decider-2b" = "decider-2b*",
          "local_semif-4b" = "SemIf-4B*", "local_nimble-9b" = "Nimble-9B*",
          "local_kev-0.8b" = "Kev-0.8B*", "local_kev-4b" = "Kev-4B*", "local_kev-9b" = "Kev-9B*")
open <- read_csv("analysis/open_models.csv", show_col_types = FALSE) %>%
  filter(model %in% names(OPEN), task %in% main$task) %>%
  transmute(model, task, ece, f1, class = "open", name = OPEN[model])
all <- bind_rows(main, open)

dotplot <- function(metric, xlab, best_low) {
  d <- all %>% filter(!is.na(.data[[metric]])) %>% mutate(v = .data[[metric]])
  ord <- d %>% group_by(name, class) %>%
    summarise(med = median(v), .groups = "drop") %>%
    arrange(if (best_low) desc(med) else med)
  d$name <- factor(d$name, levels = ord$name)
  ggplot(d, aes(v, name, colour = class)) +
    geom_point(size = 1.3, alpha = 0.55) +
    geom_point(data = ord, aes(med, name), shape = 124, size = 4, stroke = 1.2) +
    scale_colour_manual(values = c(jev = COL_JEV, local = COL_LOCAL, open = COL_OPEN, llm = COL_LLM)) +
    labs(x = xlab, y = NULL) +
    theme_dm() +
    theme(plot.margin = margin(5, 10, 5, 5),
          axis.text.y = element_text(
      colour = ifelse(ord$class == "jev", COL_JEV,
                      ifelse(ord$class %in% c("local", "open"), COL_LOCAL, "grey20"))))
}

save_fig(dotplot("ece", "ECE per evaluation task (bar = median)", TRUE),
         "fig_calibration", w = 3.4, h = 5.6)
save_fig(dotplot("f1", "Macro-F1 per evaluation task (bar = median)", FALSE),
         "fig_accuracy", w = 3.4, h = 5.6)
