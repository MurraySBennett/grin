.new_warn_collector <- function() {
  e <- new.env(parent = emptyenv())
  e$msgs <- character(0)
  e
}

.warn <- function(collector, msg) {
  collector$msgs <- c(collector$msgs, msg)
  invisible(NULL)
}
