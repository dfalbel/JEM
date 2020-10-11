bar <- function(length) {
  progress::progress_bar$new(
    format = "[:bar] :eta Loss: :loss",
    total = length
  )
}

print_config <- function(config) {
  values <- sapply(config, as.character)
  cli::cli_h1("Experiment config")
  cli::cat_bullet(paste(names(config), ":", values), bullet = "arrow_right")
  cli::cli_rule()
}