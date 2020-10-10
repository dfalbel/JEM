replay_buffer <- torch::nn_module(
  initialize = function(buffer_size) {
    self$max_size <- max_size
    self$buffer <- list()
    self$init_length <- 0
  },
  seed_buffer = function(episodes) {
    self$init_length <- length(episodes)
    self$add(episodes)
  },
  next_buffer_idx = function() {
    if (length(self$buffer) < self$max_size)
      length(self$buffer) + 1 # will add to the end of the buffer
    else
      sample.int(self$max_size, size = 1) # will randomly remove an element
  },
  add = function(episodes) {

    # allows passing a batch of tensors
    if (torch::is_torch_tensor(episodes)) {
      episodes <- torch::torch_unbind(episodes)
    }

    for (e in episodes) {
      buffer_idx <- self$next_buffer_idx()
      self$buffer[[buffer_idx]] <- e
    }
  },
  get_batch = function(n) {
    idx <- sample.int(length(self$buffer), size = n)
    self$buffer[idx]
  }
)
