suppressMessages({library(dplyr); library(readr)})
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), "theme.R"))

DISCOVERY <- c("semeval_stance", "implicit_hate", "discourse")
SHORT <- c(
  "jev" = "Jev", "local_rlcd-0.6b" = "Qwen3-0.6B-RLCD", "local_qwen3-base" = "Qwen3-0.6B base",
  "anthropic_claude-haiku-4.5" = "Haiku 4.5", "anthropic_claude-sonnet-5" = "Sonnet 5",
  "anthropic_claude-opus-5" = "Opus 5", "anthropic_claude-fable-5.1" = "Fable 5.1",
  "google_gemini-3.5-flash-lite" = "Gemini Flash-Lite", "google_gemini-3.8-flash" = "Gemini 3.8 Flash",
  "google_gemini-3.1-pro-preview" = "Gemini 3.1 Pro", "google_gemma-4-31b-it" = "Gemma 4 31B",
  "meta-llama_llama-4-scout" = "Llama 4 Scout", "meta-llama_llama-4-maverick" = "Llama 4 Maverick",
  "mistralai_mistral-medium-3-5" = "Mistral Medium", "moonshotai_kimi-k2.6" = "Kimi K2.6",
  "openai_gpt-5.6-luna" = "GPT-5.6 Luna", "openai_gpt-5.6-sol" = "GPT-5.6 Sol",
  "openai_gpt-5.6-terra" = "GPT-5.6 Terra", "openai_gpt-oss-120b" = "GPT-OSS 120B",
  "deepseek_deepseek-v3.2" = "DeepSeek V3.2", "qwen_qwen3-235b-a22b-2507" = "Qwen3 235B",
  "z-ai_glm-5.3" = "GLM 5.3")

items <- read_csv(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])),
                            "..", "analysis", "items.csv"),
                  col_types = cols_only(task = "c", model = "c", kind = "c",
                                        conf = "d", conf_status = "c")) |>
  filter(!(task %in% DISCOVERY), conf_status == "ok") |>
  mutate(class = case_when(kind == "jev" ~ "jev", kind == "local" ~ "local", TRUE ~ "llm"),
         label = factor(SHORT[model], levels = SHORT[c(
           "jev", "local_rlcd-0.6b", "local_qwen3-base",
           setdiff(names(SHORT), c("jev", "local_rlcd-0.6b", "local_qwen3-base")))]))

p <- ggplot(items, aes(conf, fill = class)) +
  geom_histogram(aes(y = after_stat(count / tapply(count, PANEL, sum)[PANEL])),
                 breaks = seq(0, 1, 0.05), colour = NA) +
  scale_fill_manual(values = c(jev = COL_JEV, local = COL_LOCAL, llm = COL_LLM)) +
  facet_wrap(~label, ncol = 5) +
  scale_x_continuous(breaks = c(0, 0.5, 1)) +
  labs(x = "Stated confidence", y = "Share of items") +
  theme_dm()

save_fig(p, "fig_conf_hist", 7.2, 6.8)
