replay_buffer <- torch::nn_module(
  initialize = function(buffer_size, dim) {
    self$max_size <- buffer_size
    self$dim <- dim
    self$buffer <- list()
    self$init_length <- 0
  },
  next_buffer_idx = function() {
    if (length(self$buffer) < self$max_size)
      length(self$buffer) + 1 # will add to the end of the buffer
    else
      sample.int(self$max_size, size = 1) # will randomly remove an element
  },
  add = function(episodes) {

    # allows passing a batch of tensors
    if (torch:::is_torch_tensor(episodes)) {
      episodes <- torch::torch_unbind(episodes)
    }

    for (e in episodes) {
      buffer_idx <- self$next_buffer_idx()
      self$buffer[[buffer_idx]] <- e
    }
  },
  get_batch = function(n, reinit_freq) {
    
    n_buffer <- rbinom(1, n, prob = 1 - reinit_freq)
    
    if (n_buffer > length(self$buffer))
      n_buffer <- length(self$buffer)
    
    n_random <- n - n_buffer
    
    tensors <- list()
    # sample from buffer
    if (n_buffer > 0) {
      idx <- sample.int(length(self$buffer), size = n_buffer)
      tensors <- append(tensors, list(torch::torch_stack(self$buffer[idx])))
    }
    
    # random sample, same size
    if (n_random > 0) {
      size <- c(n_random, self$dim)
      random_batch <- torch::torch_empty(size = size)$uniform_(-1, 1)$to(device = config::get("device"))  
      tensors <- append(tensors, list(random_batch))
    }
    
    torch::torch_cat(tensors)
  }
)
