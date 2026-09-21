library(ggplot2)

COL_JEV   <- "#E69F00"
COL_LOCAL <- "#56B4E9"
COL_LLM   <- "grey45"
COL_REF   <- "grey30"

theme_dm <- function(base_size = 9) {
  theme_minimal(base_size = base_size, base_family = "Helvetica") +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.25, colour = "grey88"),
      axis.title = element_text(size = base_size),
      strip.text = element_text(size = base_size - 1, face = "bold"),
      legend.position = "none",
      plot.title = element_blank()
    )
}

save_fig <- function(p, name, w, h) {
  dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]))
  ggsave(file.path(dir, paste0(name, ".pdf")), p, width = w, height = h,
         device = "pdf", bg = "white")
  ggsave(file.path(dir, paste0(name, ".png")), p, width = w, height = h,
         dpi = 300, bg = "white")
  cat("wrote", name, "\n")
}
