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

d <- read_csv("analysis/cell_metrics.csv", show_col_types = FALSE) %>%
  filter(split == "confirmatory", !is.na(ece)) %>%
  mutate(class = case_when(model == "jev" ~ "jev",
                           kind == "local" ~ "local",
                           TRUE ~ "llm"),
         name = SHORT[model])

ord <- d %>% group_by(name, class) %>%
  summarise(med = median(ece), .groups = "drop") %>% arrange(desc(med))
d$name <- factor(d$name, levels = ord$name)

p <- ggplot(d, aes(ece, name, colour = class)) +
  geom_point(size = 1.3, alpha = 0.55) +
  geom_point(data = ord, aes(med, name), shape = 124, size = 4, stroke = 1.2) +
  scale_colour_manual(values = c(jev = COL_JEV, local = COL_LOCAL, llm = COL_LLM)) +
  labs(x = "Expected calibration error (15 confirmatory tasks; bar = median)",
       y = NULL) +
  theme_dm() +
  theme(axis.text.y = element_text(
    colour = ifelse(ord$class == "jev", COL_JEV,
                    ifelse(ord$class == "local", COL_LOCAL, "grey20"))))

save_fig(p, "fig_calibration", w = 5.0, h = 3.6)
