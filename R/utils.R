bar <- function(length) {
  progress::progress_bar$new(
    format = "[:bar] :eta Loss: :loss",
    total = length
  )
}