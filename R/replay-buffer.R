replay_buffer <- torch::nn_module(
  initialize = function(buffer_size, dim, device) {
    self$max_size <- buffer_size
    self$dim <- dim
    self$buffer <- list()
    self$init_length <- 0
    self$device <- device
  },
  next_buffer_idx = function() {
    if (length(self$buffer) < self$max_size)
      length(self$buffer) + 1 # will add to the end of the buffer
    else
      sample.int(self$max_size, size = 1) # will randomly remove an element
  },
  add = function(episodes) {

    episodes <- episodes$x
    
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
      random_batch <- torch::torch_empty(size = size)$uniform_(-1, 1)$to(device = self$device)  
      tensors <- append(tensors, list(random_batch))
    }
    
    list(x = torch::torch_cat(tensors), y  = NULL)
  }
)

conditional_replay_buffer <- torch::nn_module(
  "conditional_replay_buffer",
  initialize = function(n_class, buffer_size, dim, device) {
    self$n_class <- n_class
    self$buffer_size <- as.integer(buffer_size/n_class)
    self$buffer <- vector(mode = "list", length = buffer_size)
    self$sizes <- rep(0, n_class)
    self$dim <- dim
    self$device <- device
  },
  id_from_y = function(id, y) {
    (y - 1)*self$buffer_size + id
  },
  next_buffer_idx = function(y) {
    y <- y$item()
    
    if (self$sizes[y] < self$buffer_size) {
      self$sizes[y] <- self$sizes[y] + 1  
      id <- self$sizes[y]
    } else {
      id <- sample.int(self$buffer_size, size = 1)
    }
      
    self$id_from_y(id, y)
  },
  add = function(episodes) {
    
    y <- episodes$y
    episodes <- episodes$x
    
    # allows passing a batch of tensors
    if (torch:::is_torch_tensor(episodes)) {
      episodes <- torch::torch_unbind(episodes)
    }
    
    if (torch:::is_torch_tensor(y)) {
      y <- torch::torch_unbind(y)
    }
    
    for (i in seq_along(episodes)) {
      id <- self$next_buffer_idx(y[[i]])
      self$buffer[[id]] <- episodes[[i]]
    }
  },
  get_batch = function(n, reinit_freq) {
    
    if (length(n) == 1)
      y <- sample.int(self$n_class, size = n, replace = TRUE)
    else
      y <- as.integer(n)
    
    batch <- vector(mode = "list", length = length(y))
    for (i in seq_along(y)) {
      r <- runif(1)
      if (r < reinit_freq || self$sizes[y[i]] <= 0) {
        batch[[i]] <- torch::torch_empty(size = self$dim)$uniform_(-1, 1)$to(device = self$device)  
      } else {
        id <- sample.int(self$sizes[y[i]], size = 1)
        id <- self$id_from_y(id, y[i])
        batch[[i]] <- self$buffer[[id]]
      }
    }
    
    list(x = torch::torch_stack(batch), y = torch::torch_tensor(y, device = self$device))
  }
)
